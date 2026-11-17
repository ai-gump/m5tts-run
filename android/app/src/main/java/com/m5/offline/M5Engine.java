package com.m5.offline;

import ai.onnxruntime.*;
import android.util.Log;
import org.json.*;
import java.io.*;
import java.nio.*;
import java.nio.file.Files;
import java.util.*;
import java.util.concurrent.atomic.AtomicBoolean;

/** CPU-only ONNX port of m5_onnx/runtime.py. Single worker owns an instance. */
public final class M5Engine implements AutoCloseable {
    public interface Progress { void update(String message); }
    public interface NoiseSource { void fill(FloatBuffer buffer) throws IOException; }
    /** 流式 PCM 接收端（AudioTrack / 文件 / 测试桩），在生成线程回调。 */
    public interface AudioSink { void write(float[] samples, int offset, int count) throws Exception; }
    private interface FrameSink { void onFrame(long[] frame, int index) throws Exception; }

    public static final List<String> LANGUAGES = Collections.unmodifiableList(
        Arrays.asList("Chinese", "English", "German", "French", "Japanese", "Korean"));
    private static final int TAIL_SAMPLES = 2400; // 100 ms，段尾修剪
    private static final int FIRST_CHUNK_FRAMES = 1; // 首块 1 帧 ≈ 80 ms 音频，压低 TTFA
    private static final int CHUNK_FRAMES = 8;       // 之后 8 帧一块；实测 16 帧块在 Kirin 990 上更慢（codec 2979→3153ms）
    private static final int PRIME_CHUNK_FRAMES = 32;

    public static final class Voice {
        public final String name, referenceText;
        final long[][] codes;
        Voice(JSONObject json) throws JSONException {
            name = json.getString("name"); referenceText = json.getString("reference_text");
            JSONArray rows = json.getJSONArray("codes");
            if (rows.length() < 1 || rows.length() > 1000 || referenceText.trim().isEmpty())
                throw new JSONException("参考音色数据无效");
            codes = new long[rows.length()][16];
            for (int i = 0; i < codes.length; i++) {
                JSONArray row = rows.getJSONArray(i);
                if (row.length() != 16) throw new JSONException("参考音色码本数量错误");
                for (int j = 0; j < 16; j++) {
                    Object raw = row.get(j); long v = row.getLong(j);
                    if (!(raw instanceof Number) || ((Number) raw).doubleValue() != v || v < 0 || v > 2047)
                        throw new JSONException("参考音色 token 错误");
                    codes[i][j] = v;
                }
            }
        }
        public String toString() { return name; }
    }

    public static final class Synthesis {
        /** 纯生成/解码耗时（不含 AudioTrack 写入阻塞）。 */
        public final double elapsedSeconds;
        /** 请求总墙钟（含播放端阻塞）。 */
        public final double wallSeconds;
        public final double audioSeconds;
        /** 首块 PCM 就绪时间；非流式或未出声时为 -1。 */
        public final double firstChunkSeconds;
        public final int segments, frames;
        Synthesis(double elapsed, double wall, double audio, double firstChunk, int count, int frames) {
            elapsedSeconds = elapsed; wallSeconds = wall; audioSeconds = audio;
            firstChunkSeconds = firstChunk; segments = count; this.frames = frames;
        }
    }

    private final OrtEnvironment env = OrtEnvironment.getEnvironment();
    private OrtSession backbone, depth, codec;
    private NpyMatrix textEmbedding, audioEmbedding;
    private Tokenizer tokenizer;
    private JSONObject meta, codecMeta;
    private String precision = "fp32";
    private final List<Voice> voices = new ArrayList<>();
    private final Map<String, DecoderSnapshot> decoderSnapshots = new HashMap<>();

    public M5Engine(File root, int threads, Progress progress) throws Exception {
        if (threads < 1 || threads > 8) throw new IllegalArgumentException("线程数需要为 1–8");
        long loadStart = System.nanoTime();
        try {
            meta = json(new File(root, "manifest.json")); codecMeta = json(new File(root, "codec_cached.json"));
            precision = meta.optString("precision", "fp32");
            if (!"m5-llama-rq16-qwen3-12hz".equals(meta.getString("architecture")) ||
                meta.getInt("layers") != 8 || meta.getInt("kv_heads") != 2 || meta.getInt("head_dim") != 64 ||
                meta.getInt("hidden_size") != 512 || meta.getInt("codebooks") != 16 ||
                meta.getInt("sample_rate") != 24000 || meta.getInt("hop_length") != 1920 ||
                meta.getInt("audio_vocab") != 2051 || codecMeta.getInt("layers") != 8 ||
                codecMeta.getInt("kv_heads") != 16 || codecMeta.getInt("head_dim") != 64)
                throw new IOException("模型架构与此版本应用不兼容");
            progress.update("正在加载文本分词器…"); tokenizer = new Tokenizer(new File(root, "tokenizer.bin"));
            textEmbedding = new NpyMatrix(new File(root, "text_embeddings.npy"));
            audioEmbedding = new NpyMatrix(new File(root, "audio_embeddings.npy"));
            if (textEmbedding.columns != 512 || textEmbedding.rows != 129992 ||
                audioEmbedding.columns != 512 || audioEmbedding.rows != 16 * 2051)
                throw new IOException("Embedding 维度与模型不匹配");
            JSONArray list = json(new File(root, "voices.json")).getJSONArray("voices");
            for (int i = 0; i < list.length(); i++) voices.add(new Voice(list.getJSONObject(i)));
            if (voices.isEmpty()) throw new IOException("模型包没有音色");
            try (OrtSession.SessionOptions options = new OrtSession.SessionOptions()) {
                options.setIntraOpNumThreads(threads); options.setInterOpNumThreads(1);
                options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
                options.addConfigEntry("session.intra_op.allow_spinning", "0");
                options.addConfigEntry("session.inter_op.allow_spinning", "0");
                progress.update("正在加载语音生成模型…");
                backbone = env.createSession(new File(root, "backbone.onnx").getPath(), options);
                depth = env.createSession(new File(root, "depth_frame.onnx").getPath(), options);
                progress.update("正在加载声音解码器…");
                codec = env.createSession(new File(root, "codec_decoder_cached.onnx").getPath(), options);
            }
        } catch (Exception | OutOfMemoryError e) { close(); throw e; }
        Log.i("M5Perf", String.format(Locale.ROOT, "[M5_LOAD] threads=%d load_ms=%.0f precision=%s",
            threads, (System.nanoTime() - loadStart) / 1e6, precision));
    }

    public List<Voice> voices() { return Collections.unmodifiableList(voices); }
    public String precision() { return precision; }
    private static String voiceKey(Voice voice) {
        return voice.name + '\n' + voice.referenceText + '\n' + Arrays.deepHashCode(voice.codes);
    }
    /** 把参考音色的解码器状态预先 prime 并缓存；之后每段/每请求直接恢复。 */
    public void prepareVoice(Voice voice, Progress progress) throws Exception {
        if (voice == null) throw new IllegalArgumentException("请选择音色");
        String key = voiceKey(voice);
        if (decoderSnapshots.containsKey(key)) return;
        progress.update("正在预热音色…");
        long start = System.nanoTime();
        try (CodecState state = new CodecState(null)) {
            for (int offset = 0; offset < voice.codes.length; offset += PRIME_CHUNK_FRAMES) {
                state.run(voice.codes, offset, Math.min(PRIME_CHUNK_FRAMES, voice.codes.length - offset));
            }
            decoderSnapshots.put(key, snapshot(state, voice.codes.length));
        }
        Log.i("M5Perf", String.format(Locale.ROOT,
            "[M5_PREPARE] voice_ms=%.0f ref_frames=%d", (System.nanoTime() - start) / 1e6, voice.codes.length));
    }
    public boolean isVoicePrepared(Voice voice) {
        return voice != null && decoderSnapshots.containsKey(voiceKey(voice));
    }
    public static JSONObject json(File path) throws IOException, JSONException {
        return new JSONObject(new String(Files.readAllBytes(path.toPath()), java.nio.charset.StandardCharsets.UTF_8));
    }

    public static String conditioned(String reference, String text, String language) {
        if (!LANGUAGES.contains(language)) throw new IllegalArgumentException("不支持的语言：" + language);
        String combined = reference.trim() + text.trim();
        for (String token : new String[]{"<|caption_begin|>", "<|caption_end|>", "<|accent_begin|>",
            "<|accent_end|>", "<|speaker_start|>", "<|speaker_end|>", "<|lang_begin|>", "<|lang_end|>"})
            if (combined.contains(token)) throw new IllegalArgumentException("文本包含保留的控制标记");
        return "<|caption_begin|><|caption_end|><|accent_begin|><|accent_end|>" +
            "<|speaker_start|>None<|speaker_end|><|lang_begin|>" + language + "<|lang_end|>" + combined;
    }
    public static String conditioned(String reference, String text) { return conditioned(reference, text, "Chinese"); }

    public Synthesis synthesize(String text, int voiceIndex, long seed, File output,
                                AtomicBoolean cancelled, Progress progress) throws Exception {
        if (voiceIndex < 0 || voiceIndex >= voices.size()) throw new IllegalArgumentException("请选择音色");
        return synthesize(text, voices.get(voiceIndex), seed, output, cancelled, progress);
    }
    public Synthesis synthesize(String text, Voice voice, long seed, File output,
                                AtomicBoolean cancelled, Progress progress) throws Exception {
        Random random = new Random(seed);
        return synthesize(text, voice, "Chinese", output, cancelled, progress, buffer -> {
            for (int i = 0; i < buffer.capacity(); i++)
                buffer.put(i, Math.max((float) -Math.log(1.0 - random.nextDouble()), Float.MIN_NORMAL));
        }, null);
    }
    /** 带语言与流式回调的入口：sink 为 null 时退回“整段生成后再解码”。 */
    public Synthesis synthesize(String text, Voice voice, String language, long seed, File output,
                                AtomicBoolean cancelled, Progress progress, AudioSink sink) throws Exception {
        Random random = new Random(seed);
        return synthesize(text, voice, language, output, cancelled, progress, buffer -> {
            for (int i = 0; i < buffer.capacity(); i++)
                buffer.put(i, Math.max((float) -Math.log(1.0 - random.nextDouble()), Float.MIN_NORMAL));
        }, sink);
    }

    public Synthesis synthesize(String text, int voiceIndex, File output, AtomicBoolean cancelled,
                                Progress progress, NoiseSource noiseSource) throws Exception {
        if (voiceIndex < 0 || voiceIndex >= voices.size()) throw new IllegalArgumentException("请选择音色");
        return synthesize(text, voices.get(voiceIndex), output, cancelled, progress, noiseSource);
    }
    public Synthesis synthesize(String text, Voice voice, File output, AtomicBoolean cancelled,
                                Progress progress, NoiseSource noiseSource) throws Exception {
        return synthesize(text, voice, "Chinese", output, cancelled, progress, noiseSource, null);
    }

    public Synthesis synthesize(String text, Voice voice, String language, File output, AtomicBoolean cancelled,
                                Progress progress, NoiseSource noiseSource, AudioSink sink) throws Exception {
        if (voice == null) throw new IllegalArgumentException("请选择音色");
        if (!LANGUAGES.contains(language)) throw new IllegalArgumentException("不支持的语言：" + language);
        List<String> segments = splitText(text);
        long start = System.nanoTime();
        double duration, firstChunk = -1;
        long sinkNanos = 0; int framesTotal = 0;
        boolean success = false;
        try (WavWriter wav = new WavWriter(output)) {
            for (int i = 0; i < segments.size(); i++) {
                check(cancelled);
                final String prefix = "第 " + (i + 1) + "/" + segments.size() + " 段：";
                long segStart = System.nanoTime();
                double segAudioStart = wav.duration();
                if (sink != null) {
                    try (StreamingDecoder decoder = new StreamingDecoder(
                            voice, wav, sink, cancelled, start)) {
                        long[][] generated = generate(segments.get(i), voice, language, cancelled,
                            message -> progress.update(prefix + message), noiseSource, decoder);
                        decoder.finish();
                        framesTotal += generated.length;
                        sinkNanos += decoder.sinkNanos();
                        if (firstChunk < 0 && decoder.firstChunkSeconds() >= 0) firstChunk = decoder.firstChunkSeconds();
                        logSegment(i, segments.size(), generated.length, segStart, segAudioStart, wav,
                            decoder.sinkNanos(), decoder.decodeNanos());
                    }
                } else {
                    long[][] generated = generate(segments.get(i), voice, language, cancelled,
                        message -> progress.update(prefix + message), noiseSource, null);
                    progress.update(prefix + "正在转换成声音…");
                    decode(voice.codes, generated, wav, cancelled);
                    framesTotal += generated.length;
                    logSegment(i, segments.size(), generated.length, segStart, segAudioStart, wav, 0, 0);
                }
            }
            duration = wav.duration();
            if (duration <= 0) throw new IOException("模型没有生成有效音频");
            success = true;
        } finally { if (!success) Files.deleteIfExists(output.toPath()); }
        long end = System.nanoTime();
        double wall = (end - start) / 1e9;
        double compute = Math.max(0, end - start - sinkNanos) / 1e9;
        if (firstChunk >= 0) Log.i("M5Perf", String.format(Locale.ROOT,
            "[M5_TTFA] first_chunk_ms=%.0f", firstChunk * 1000));
        return new Synthesis(compute, wall, duration, firstChunk, segments.size(), framesTotal);
    }

    private void logSegment(int index, int total, int frames, long segStart, double audioStart,
                            WavWriter wav, long sinkNanos, long decodeNanos) {
        double gen = (System.nanoTime() - segStart - sinkNanos) / 1e9;
        double audio = wav.duration() - audioStart;
        double llm = Math.max(0, gen - decodeNanos / 1e9);
        Log.i("M5Perf", String.format(Locale.ROOT,
            "[M5_SEG] index=%d/%d frames=%d audio_ms=%.0f gen_ms=%.0f llm_ms=%.0f codec_ms=%.0f rtf=%.3f",
            index + 1, total, frames, audio * 1000, gen * 1000, llm * 1000, decodeNanos / 1e6,
            audio > 0 ? gen / audio : 0));
    }

    private long[][] generate(String text, Voice voice, String language, AtomicBoolean cancelled,
                              Progress progress, NoiseSource source, FrameSink sink) throws Exception {
        long genStart = System.nanoTime();
        long[] ids = tokenizer.encode(conditioned(voice.referenceText, text, language));
        int maxFrames = 500, length = ids.length + voice.codes.length;
        Log.i("M5Perf", String.format(Locale.ROOT,
            "[M5_TOKENIZE] ids=%d ref_frames=%d prefix=%d tokenize_ms=%.1f",
            ids.length, voice.codes.length, length, (System.nanoTime() - genStart) / 1e6));
        if (length + maxFrames > meta.getInt("max_sequence")) throw new IOException("文本或参考音色过长，请缩短后重试");
        FloatBuffer prefix = floats(length * 512);
        for (long id : ids) textEmbedding.copyRow((int) id, prefix);
        for (long[] row : voice.codes) {
            float[] sum = new float[512];
            for (int j = 0; j < 16; j++) audioEmbedding.addRow((int) row[j] + 1 + j * 2051, sum);
            prefix.put(sum);
        }
        prefix.rewind();
        List<long[]> generated = new ArrayList<>();
        OrtSession.Result previousBackbone = null, previousDepth = null;
        FloatBuffer noise = floats(16 * 2051);
        try (OnnxTensor initialHidden = OnnxTensor.createTensor(env, prefix, new long[]{1, length, 512});
             OnnxTensor initialPast = zeros(new long[]{8, 2, 1, 2, 0, 64});
             OnnxTensor noiseTensor = OnnxTensor.createTensor(env, noise, new long[]{16, 1, 2051})) {
            for (int step = 0; step < maxFrames; step++) {
                check(cancelled);
                long[] positions = step == 0 ? new long[length] : new long[]{length + step - 1L};
                if (step == 0) for (int i = 0; i < length; i++) positions[i] = i;
                try (OnnxTensor position = longs(positions, new long[]{1, positions.length})) {
                    Map<String, OnnxTensor> feed = new HashMap<>();
                    feed.put("hidden", previousDepth == null ? initialHidden : tensor(previousDepth, "next_hidden"));
                    feed.put("past", previousBackbone == null ? initialPast : tensor(previousBackbone, "present"));
                    feed.put("positions", position);
                    OrtSession.Result next = backbone.run(feed);
                    if (step == 0) Log.i("M5Perf", String.format(Locale.ROOT,
                        "[M5_PREFILL] prefill_ms=%.1f positions=%d",
                        (System.nanoTime() - genStart) / 1e6, positions.length));
                    if (previousBackbone != null) previousBackbone.close(); previousBackbone = next;
                }
                if (previousDepth != null) { previousDepth.close(); previousDepth = null; }
                source.fill(noise);
                try (OnnxTensor allowEos = OnnxTensor.createTensor(env, new boolean[]{step > meta.getInt("min_frames")})) {
                    Map<String, OnnxTensor> feed = new HashMap<>();
                    feed.put("hidden", tensor(previousBackbone, "last_hidden"));
                    feed.put("noise", noiseTensor); feed.put("allow_eos", allowEos);
                    previousDepth = depth.run(feed);
                }
                if (step == 0) Log.i("M5Perf", String.format(Locale.ROOT,
                    "[M5_FIRST_FRAME] first_frame_ms=%.1f",
                    (System.nanoTime() - genStart) / 1e6));
                LongBuffer codes = tensor(previousDepth, "codes").getLongBuffer();
                long[] frame = new long[16]; boolean eos = true;
                for (int j = 0; j < 16; j++) { long v = codes.get(j); eos &= v == 0; frame[j] = Math.max(0, Math.min(2047, v - 1)); }
                if (eos) {
                    if (generated.isEmpty()) throw new IOException("模型提前结束，没有生成语音");
                    return generated.toArray(new long[0][]);
                }
                generated.add(frame);
                if (sink != null) sink.onFrame(frame, generated.size());
                if (step % 8 == 0) progress.update(String.format(Locale.CHINA, "已生成 %.1f 秒语音", generated.size() * 0.08));
            }
            throw new IOException("此段超过 40 秒生成上限，请缩短文本后重试");
        } finally {
            if (previousBackbone != null) previousBackbone.close();
            if (previousDepth != null) previousDepth.close();
        }
    }

    /** 边生成边解码：每 4 帧进解码器，吐出时保留末尾 100 ms，段尾丢弃。 */
    private final class StreamingDecoder implements FrameSink, AutoCloseable {
        private final CodecState state;
        private final WavWriter wav;
        private final AudioSink sink;
        private final AtomicBoolean cancelled;
        private final long startNanos;
        private final List<long[]> buffered = new ArrayList<>();
        private float[] pending = new float[16384];
        private int pendingCount, chunkTarget = FIRST_CHUNK_FRAMES;
        private long sinkNanos, decodeNanos;
        private boolean firstDecodeLogged;
        private double firstChunkSeconds = -1;
        private boolean closed;

        StreamingDecoder(Voice voice, WavWriter wav, AudioSink sink,
                         AtomicBoolean cancelled, long startNanos) throws Exception {
            this.wav = wav; this.sink = sink; this.cancelled = cancelled; this.startNanos = startNanos;
            long[][] reference = voice.codes;
            String key = voiceKey(voice);
            DecoderSnapshot cached = decoderSnapshots.get(key);
            long primeStart = System.nanoTime();
            state = new CodecState(cached);
            try {
                if (cached == null) {
                    for (int start = 0; start < reference.length; start += PRIME_CHUNK_FRAMES) {
                        check(cancelled);
                        state.run(reference, start, Math.min(PRIME_CHUNK_FRAMES, reference.length - start));
                    }
                    decoderSnapshots.put(key, snapshot(state, reference.length));
                }
            } catch (Exception e) { state.close(); throw e; }
            Log.i("M5Perf", String.format(Locale.ROOT,
                "[M5_PRIME] prime_ms=%.0f ref_frames=%d source=%s",
                (System.nanoTime() - primeStart) / 1e6, reference.length, cached == null ? "compute" : "snapshot"));
        }

        public void onFrame(long[] frame, int index) throws Exception {
            check(cancelled);
            buffered.add(frame);
            if (buffered.size() >= chunkTarget) flushChunk();
        }

        private void flushChunk() throws Exception {
            if (buffered.isEmpty()) return;
            long[][] chunk = buffered.toArray(new long[0][]);
            buffered.clear();
            long decodeStart = System.nanoTime();
            FloatBuffer pcm = state.run(chunk, 0, chunk.length);
            long decodeMs = System.nanoTime() - decodeStart;
            decodeNanos += decodeMs;
            if (!firstDecodeLogged) {
                firstDecodeLogged = true;
                chunkTarget = CHUNK_FRAMES;
                Log.i("M5Perf", String.format(Locale.ROOT,
                    "[M5_DEC_FIRST] decode_ms=%.1f frames=%d", decodeMs / 1e6, chunk.length));
            }
            int count = pcm.remaining();
            if (pendingCount + count > pending.length)
                pending = Arrays.copyOf(pending, Math.max(pending.length * 2, pendingCount + count));
            pcm.get(pending, pendingCount, count);
            pendingCount += count;
            emitAvailable();
        }

        private void emitAvailable() throws Exception {
            if (firstChunkSeconds < 0) {
                // 首块只有 1 帧（1920 样本）时不够 2400 的保留阈值：直接整块发出，
                // 后续仍按“保留末尾 2400 样本”推进；总量仍是 total-2400。
                if (pendingCount >= 1920) {
                    firstChunkSeconds = (System.nanoTime() - startNanos) / 1e9;
                    if (sink != null) {
                        long t0 = System.nanoTime();
                        sink.write(pending, 0, pendingCount);
                        sinkNanos += System.nanoTime() - t0;
                    }
                    wav.append(FloatBuffer.wrap(pending, 0, pendingCount), pendingCount);
                    pendingCount = 0;
                }
                return;
            }
            int emit = pendingCount - TAIL_SAMPLES;
            if (emit <= 0) return;
            if (sink != null) {
                long t0 = System.nanoTime();
                sink.write(pending, 0, emit);
                sinkNanos += System.nanoTime() - t0;
            }
            wav.append(FloatBuffer.wrap(pending, 0, emit), emit);
            System.arraycopy(pending, emit, pending, 0, TAIL_SAMPLES);
            pendingCount = TAIL_SAMPLES;
        }

        void finish() throws Exception {
            flushChunk();
            if (pendingCount < TAIL_SAMPLES) throw new IOException("声音解码器返回的长度不足");
            emitAvailable();
            if (pendingCount != TAIL_SAMPLES) throw new IOException("声音解码器返回的长度不足");
            pendingCount = 0; // 丢弃段尾 100 ms
            close();
        }

        long sinkNanos() { return sinkNanos; }
        long decodeNanos() { return decodeNanos; }
        double firstChunkSeconds() { return firstChunkSeconds; }
        public void close() {
            if (closed) return; closed = true;
            state.close();
        }
    }

    private void decode(long[][] reference, long[][] generated, WavWriter wav, AtomicBoolean cancelled) throws Exception {
        try (CodecState state = new CodecState(null)) {
            for (int start = 0; start < reference.length; start += 16) {
                check(cancelled); state.run(reference, start, Math.min(16, reference.length - start));
            }
            int remaining = generated.length * 1920 - 2400; // Same 100 ms tail trim as Python.
            for (int start = 0; start < generated.length; start += 32) {
                check(cancelled);
                FloatBuffer pcm = state.run(generated, start, Math.min(32, generated.length - start));
                int count = Math.max(0, Math.min(remaining, pcm.remaining()));
                wav.append(pcm, count); remaining -= count;
            }
            if (remaining > 0) throw new IOException("声音解码器返回的长度不足");
        }
    }

    private final class CodecState implements AutoCloseable {
        private final List<OnnxTensor> initial = new ArrayList<>();
        private final Map<String, OnnxTensor> restored;
        private OrtSession.Result previous;
        private int offset;
        CodecState(DecoderSnapshot snapshot) throws Exception {
            restored = snapshot == null ? null : snapshot.states;
            offset = snapshot == null ? 0 : snapshot.frames;
            try {
                initial.add(zeros(new long[]{8, 2, 1, 16, 0, 64}));
                JSONArray states = codecMeta.getJSONArray("states");
                for (int i = 0; i < states.length(); i++) {
                    JSONArray a = states.getJSONObject(i).getJSONArray("shape"); long[] shape = new long[a.length()];
                    for (int j = 0; j < shape.length; j++) shape[j] = a.getLong(j);
                    initial.add(zeros(shape));
                }
            } catch (Exception e) { close(); throw e; }
        }
        FloatBuffer run(long[][] frames, int start, int count) throws Exception {
            long[] flat = new long[16 * count];
            for (int c = 0; c < 16; c++) for (int i = 0; i < count; i++) flat[c * count + i] = frames[start + i][c];
            try (OnnxTensor codes = longs(flat, new long[]{1, 16, count});
                 OnnxTensor position = longs(new long[]{offset}, new long[]{1})) {
                Map<String, OnnxTensor> feed = new HashMap<>();
                feed.put("codes", codes); feed.put("offset", position);
                feed.put("past", previous == null ? stateTensor("present", 0) : tensor(previous, "present"));
                for (int i = 1; i < initial.size(); i++)
                    feed.put("conv_" + (i - 1), previous == null
                        ? stateTensor("next_conv_" + (i - 1), i)
                        : tensor(previous, "next_conv_" + (i - 1)));
                OrtSession.Result next = codec.run(feed);
                if (previous != null) previous.close(); previous = next; offset += count;
                return tensor(previous, "audio").getFloatBuffer();
            }
        }
        private OnnxTensor stateTensor(String name, int initialIndex) {
            return restored != null ? restored.get(name) : initial.get(initialIndex);
        }
        public void close() { if (previous != null) previous.close(); for (OnnxTensor t : initial) t.close(); }
    }

    private final class DecoderSnapshot {
        final Map<String, OnnxTensor> states;
        final int frames;
        DecoderSnapshot(Map<String, OnnxTensor> states, int frames) { this.states = states; this.frames = frames; }
        void close() { for (OnnxTensor tensor : states.values()) try { tensor.close(); } catch (Exception ignored) { } }
    }

    private DecoderSnapshot snapshot(CodecState state, int frames) throws Exception {
        Map<String, OnnxTensor> copy = new LinkedHashMap<>();
        for (String name : codecStateNames()) {
            OnnxTensor source = tensor(state.previous, name);
            FloatBuffer values = source.getFloatBuffer();
            int count = values.remaining();
            FloatBuffer target = ByteBuffer.allocateDirect(count * 4).order(ByteOrder.nativeOrder()).asFloatBuffer();
            target.put(values); target.rewind();
            copy.put(name, OnnxTensor.createTensor(env, target, source.getInfo().getShape()));
        }
        return new DecoderSnapshot(copy, frames);
    }

    private List<String> codecStateNames() throws JSONException {
        List<String> names = new ArrayList<>();
        names.add("present");
        JSONArray states = codecMeta.getJSONArray("states");
        for (int i = 0; i < states.length(); i++) names.add("next_conv_" + i);
        return names;
    }

    private static OnnxTensor tensor(OrtSession.Result result, String name) throws IOException {
        OnnxValue value = result.get(name).orElseThrow(() -> new IOException("模型缺少输出：" + name));
        if (!(value instanceof OnnxTensor)) throw new IOException("模型输出类型错误：" + name);
        return (OnnxTensor) value;
    }
    private static FloatBuffer floats(int count) { return ByteBuffer.allocateDirect(count * 4).order(ByteOrder.nativeOrder()).asFloatBuffer(); }
    private OnnxTensor zeros(long[] shape) throws OrtException {
        long count = 1; for (long d : shape) count *= d;
        return OnnxTensor.createTensor(env, floats(Math.toIntExact(count)), shape);
    }
    private OnnxTensor longs(long[] values, long[] shape) throws OrtException {
        LongBuffer buffer = ByteBuffer.allocateDirect(values.length * 8).order(ByteOrder.nativeOrder()).asLongBuffer();
        buffer.put(values).rewind(); return OnnxTensor.createTensor(env, buffer, shape);
    }
    private static void check(AtomicBoolean cancelled) throws InterruptedIOException {
        if (cancelled.get() || Thread.currentThread().isInterrupted()) throw new InterruptedIOException("已停止生成");
    }
    public static List<String> splitText(String text) {
        if (text == null || text.trim().isEmpty()) throw new IllegalArgumentException("请输入要合成的文字");
        if (text.codePointCount(0, text.length()) > 2000) throw new IllegalArgumentException("一次最多输入 2000 字");
        List<String> result = new ArrayList<>(); StringBuilder part = new StringBuilder(); int count = 0;
        for (int offset = 0; offset < text.length();) {
            int cp = text.codePointAt(offset); offset += Character.charCount(cp); part.appendCodePoint(cp); count++;
            if (count >= 80 || "。！？!?；;\n".indexOf(cp) >= 0) {
                if (!part.toString().trim().isEmpty()) result.add(part.toString().trim()); part.setLength(0); count = 0;
            }
        }
        if (!part.toString().trim().isEmpty()) result.add(part.toString().trim());
        if (result.isEmpty()) throw new IllegalArgumentException("请输入有效文字"); return result;
    }
    public void close() {
        for (DecoderSnapshot snapshot : decoderSnapshots.values()) snapshot.close();
        decoderSnapshots.clear();
        for (OrtSession session : new OrtSession[]{codec, depth, backbone})
            if (session != null) try { session.close(); } catch (OrtException ignored) { }
        for (NpyMatrix matrix : new NpyMatrix[]{textEmbedding, audioEmbedding})
            if (matrix != null) try { matrix.close(); } catch (IOException ignored) { }
        codec = null; depth = null; backbone = null; textEmbedding = null; audioEmbedding = null; tokenizer = null;
    }
}

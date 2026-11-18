import Foundation
import os

public let m5log = OSLog(subsystem: "com.aigump.m5demo", category: "M5Perf")

func m5perf(_ message: String) {
    os_log("%{public}s", log: m5log, message)
}

// MARK: - 采样统计

public struct Synthesis {
    public let elapsedSeconds: Double      // 生成/解码（不含播放阻塞）
    public let wallSeconds: Double
    public let audioSeconds: Double
    public let firstChunkSeconds: Double   // -1 = 非流式
    public let segments: Int
    public let frames: Int
}

/// 流式 PCM 接收端（AVAudioPlayerNode 在这里吃数据）
public protocol AudioSink: AnyObject {
    func write(_ samples: UnsafePointer<Float>, count: Int)
}

// MARK: - ORT 运行时抽象（真实实现依赖 OnnxRuntime xcframework；缺失时给出可诊断错误）

public protocol ORTRunner: AnyObject {
    func runBackbone(hidden: UnsafePointer<Float>, hiddenCount: Int,
                     past: UnsafePointer<Float>?, pastFrames: Int,
                     positions: [Int64],
                     outLastHidden: UnsafeMutablePointer<Float>,
                     outPresent: UnsafeMutablePointer<Float>) throws -> Int   // 返回 present 帧数
    func runDepth(lastHidden: UnsafePointer<Float>, noise: UnsafePointer<Float>,
                  allowEos: Bool, outCodes: UnsafeMutablePointer<Int64>,
                  outNextHidden: UnsafeMutablePointer<Float>) throws
    /// 返回 (audio样本数, present帧数)；nextConvs 写入调用方缓冲
    func runCodec(codes: UnsafePointer<Int64>, frames: Int, offset: Int,
                  past: UnsafePointer<Float>?, pastFrames: Int,
                  convs: UnsafePointer<Float>?,
                  outAudio: UnsafeMutablePointer<Float>,
                  outPresent: UnsafeMutablePointer<Float>,
                  outConvs: UnsafeMutablePointer<Float>) throws -> (Int, Int)
    var codecStateBytes: Int { get }
}

public enum M5RunnerError: Error, LocalizedError {
    case frameworkMissing
    public var errorDescription: String? {
        "缺少 OnnxRuntime xcframework：先运行 ios/scripts/download_framework.sh（或参考 ios/README.md 的离线获取方式）"
    }
}

// MARK: - 引擎

public final class M5Engine {
    public static let tailSamples = 2400      // 100ms 段尾修剪
    static let firstChunkFrames = 1           // 首块 1 帧（TTFA）
    static let chunkFrames = 8                // 后续 8 帧一块
    static let primeChunkFrames = 32
    static let maxFrames = 500

    public struct Meta {
        var layers = 8, kvHeads = 2, headDim = 64, hidden = 512
        var codebooks = 16, audioVocab = 2051, minFrames = 6, maxSequence = 2048
    }

    public let model: ModelDirectory
    public let precision: String
    let meta = Meta()
    let tokenizer: M5Tokenizer
    let textEmbedding: NpyMatrix
    let audioEmbedding: NpyMatrix
    private let runner: ORTRunner
    private var decoderSnapshots: [String: [Float]] = [:]   // voiceKey -> 22 状态张量平铺
    private var snapshotFrames: [String: Int] = [:]

    public init(model: ModelDirectory, runner: ORTRunner) throws {
        self.model = model
        self.runner = runner
        precision = model.precision
        let t0 = Date()
        tokenizer = try M5Tokenizer(fileURL: model.url.appendingPathComponent("tokenizer.bin"))
        textEmbedding = try NpyMatrix(url: model.url.appendingPathComponent("text_embeddings.npy"), expectRows: 129992, expectColumns: 512)
        audioEmbedding = try NpyMatrix(url: model.url.appendingPathComponent("audio_embeddings.npy"), expectRows: 16 * 2051, expectColumns: 512)
        let load = Date().timeIntervalSince(t0) * 1000
        m5perf(String(format: "[M5_LOAD] load_ms=%.0f precision=%@", load, precision))
    }

    static func voiceKey(_ v: M5Voice) -> String { v.name + "\n" + v.referenceText }

    // MARK: 音色预热（prime -> 快照）

    public func prepareVoice(_ voice: M5Voice) throws {
        let key = Self.voiceKey(voice)
        guard decoderSnapshots[key] == nil else { return }
        let t0 = Date()
        var past: [Float]? = nil; var convs = [Float](repeating: 0, count: runner.codecStateBytes / 4)
        var offset = 0
        var scratch = [Float](repeating: 0, count: 64 * 1920)
        var present = [Float](repeating: 0, count: 8 * 2 * 16 * 64 * (voice.frames + 16))
        while offset < voice.frames {
            let n = min(Self.primeChunkFrames, voice.frames - offset)
            var codes = [Int64](repeating: 0, count: 16 * n)
            for f in 0..<n { for b in 0..<16 { codes[b * n + f] = Int64(voice.code(offset + f, b)) } }
            let audioCount = try runner.runCodec(codes: codes, frames: n, offset: offset,
                                                 past: past.flatMap { $0.withUnsafeBufferPointer { $0.baseAddress } }, pastFrames: offset,
                                                 convs: convs.withUnsafeBufferPointer { $0.baseAddress! },
                                                 outAudio: &scratch, outPresent: &present, outConvs: &convs)
            _ = audioCount
            past = present
            // present 帧数 = pastFrames + n；为下一块准备精确缓冲
            let newLen = 8 * 2 * 16 * 64 * (offset + n)
            past = Array(past![0..<min(newLen, past!.count)])
            present = [Float](repeating: 0, count: 8 * 2 * 16 * 64 * (voice.frames + 16))
            offset += n
        }
        decoderSnapshots[key] = convs
        snapshotFrames[key] = voice.frames
        m5perf(String(format: "[M5_PREPARE] voice_ms=%.0f ref_frames=%d", Date().timeIntervalSince(t0) * 1000, voice.frames))
    }

    // MARK: 合成（流式回调）

    public func synthesize(text: String, voice: M5Voice, language: M5Language,
                           output: URL, cancel: inout Bool, progress: (String) -> Void,
                           sink: AudioSink?) throws -> Synthesis {
        let segments = try splitText(text)
        let start = Date()
        var totalFrames = 0, sinkNanos = 0.0
        var firstChunk = -1.0
        let wav = try WavWriter(url: output)
        defer { try? wav.close() }

        for (i, segment) in segments.enumerated() {
            if cancel { throw M5EngineError.emptyText }   // 复用中断路径，上层展示"已停止"
            let prefix = "第 \(i+1)/\(segments.count) 段："
            let segStart = Date(); let audioBefore = wav.duration
            let decoder = StreamingDecoder(engine: self, voice: voice, wav: wav, sink: sink, start: start)
            var frames = 0
            try generate(segment: segment, voice: voice, language: language, cancel: &cancel,
                         progress: { progress(prefix + $0) }) { frame, _ in
                frames += 1
                try decoder.push(frame)
            }
            try decoder.finish()
            totalFrames += frames
            if decoder.firstChunkSeconds > 0, firstChunk < 0 { firstChunk = decoder.firstChunkSeconds }
            sinkNanos += decoder.sinkSeconds
            let gen = Date().timeIntervalSince(segStart) - decoder.sinkSeconds
            let audio = wav.duration - audioBefore
            m5perf(String(format: "[M5_SEG] index=%d/%d frames=%d audio_ms=%.0f gen_ms=%.0f llm_ms=%.0f codec_ms=%.0f rtf=%.3f",
                          i + 1, segments.count, frames, audio * 1000, gen * 1000,
                          (gen - decoder.codecSeconds) * 1000, decoder.codecSeconds * 1000,
                          audio > 0 ? gen / audio : 0))
        }
        guard wav.duration > 0 else { throw M5EngineError.badModel("模型没有生成有效音频") }
        if firstChunk >= 0 { m5perf(String(format: "[M5_TTFA] first_chunk_ms=%.0f", firstChunk * 1000)) }
        let wall = Date().timeIntervalSince(start)
        return Synthesis(elapsedSeconds: max(0, wall - sinkNanos), wallSeconds: wall,
                         audioSeconds: wav.duration, firstChunkSeconds: firstChunk,
                         segments: segments.count, frames: totalFrames)
    }

    // MARK: 逐帧生成（backbone prefill -> depth 循环）

    private func generate(segment: String, voice: M5Voice, language: M5Language,
                          cancel: inout Bool, progress: (String) -> Void,
                          onFrame: (UnsafePointer<Int64>, Int) throws -> Void) throws {
        let ids = tokenizer.encode(try conditioned(reference: voice.referenceText, text: segment, language: language))
        let length = ids.count + voice.frames
        guard length + Self.maxFrames <= meta.maxSequence else { throw M5EngineError.badModel("文本或参考音色过长") }

        // prefix = 文本 embedding + 参考音频 embedding 求和折叠
        var prefix = [Float](repeating: 0, count: length * 512)
        prefix.withUnsafeMutableBufferPointer { buf in
            var at = 0
            for id in ids { textEmbedding.copyRow(Int(id), into: buf.baseAddress! + at); at += 512 }
            for f in 0..<voice.frames {
                let row = buf.baseAddress! + at
                for b in 0..<16 { audioEmbedding.addRow(voice.code(f, b) + 1 + b * meta.audioVocab, into: row) }
                at += 512
            }
        }
        let prefillStart = Date()
        var present = [Float](repeating: 0, count: 8 * 2 * 2 * 64 * (length + Self.maxFrames + 8))
        var lastHidden = [Float](repeating: 0, count: 512)
        var nextHidden = [Float](repeating: 0, count: 512)
        var codes = [Int64](repeating: 0, count: 16)

        var step = 0
        while step < Self.maxFrames {
            if cancel { throw M5EngineError.emptyText }
            var positions = [Int64](repeating: 0, count: step == 0 ? length : 1)
            if step == 0 { for i in 0..<length { positions[i] = Int64(i) } } else { positions[0] = Int64(length + step - 1) }
            let inHidden = step == 0 ? prefix : nextHidden
            let pastCount = step == 0 ? 0 : length + step - 1
            let presentFrames = try inHidden.withUnsafeBufferPointer { ih in
                try positions.withUnsafeBufferPointer { pos in
                    try lastHidden.withUnsafeMutableBufferPointer { lh in
                        try present.withUnsafeMutableBufferPointer { pr in
                            var pastPtr: UnsafePointer<Float>? = nil
                            if step > 0 { pastPtr = UnsafePointer(pr.baseAddress!) }  // present 缓冲即 past（含全历史，从头读）
                            return try runner.runBackbone(hidden: ih.baseAddress!, hiddenCount: inHidden.count,
                                                          past: pastPtr, pastFrames: pastCount,
                                                          positions: Array(pos), outLastHidden: lh.baseAddress!, outPresent: pr.baseAddress!)
                        }
                    }
                }
            }
            if step == 0 { m5perf(String(format: "[M5_PREFILL] prefill_ms=%.1f positions=%d", Date().timeIntervalSince(prefillStart) * 1000, length)) }
            // present 的布局为 [8,2,2,frames,64]（kv=2 heads）；下一块 past 从同缓冲头部继续读
            _ = presentFrames

            // depth：gumbel 噪声 + 采样
            var noise = [Float](repeating: 0, count: 16 * meta.audioVocab)
            for i in 0..<noise.count {
                let u = Double.random(in: 1e-9..<1.0)
                let g = Float(-log(1.0 - u))
                noise[i] = (g.isFinite && g > 0) ? g : Float.leastNormalMagnitude
            }
            try noise.withUnsafeBufferPointer { nz in
                try lastHidden.withUnsafeBufferPointer { lh in
                    try codes.withUnsafeMutableBufferPointer { c in
                        try nextHidden.withUnsafeMutableBufferPointer { nh in
                            try runner.runDepth(lastHidden: lh.baseAddress!, noise: nz.baseAddress!, allowEos: step > meta.minFrames,
                                                outCodes: c.baseAddress!, outNextHidden: nh.baseAddress!)
                        }
                    }
                }
            }
            if step == 0 { m5perf(String(format: "[M5_FIRST_FRAME] first_frame_ms=%.1f", Date().timeIntervalSince(prefillStart) * 1000)) }
            var eos = true
            for b in 0..<16 where codes[b] != 0 { eos = false }
            if eos {
                if step == 0 { throw M5EngineError.badModel("模型提前结束，没有生成语音") }
                return
            }
            try codes.withUnsafeBufferPointer { c in try onFrame(c.baseAddress!, Int(step)) }
            if step % 8 == 0 {
                progress(String(format: "已生成 %.1f 秒语音", Double(step + 1) * 0.08))
            }
            // 下一帧输入 hidden 用 depth 输出；prefix 不再用
            if step == 0 { prefix = nextHidden }   // 省内存：prefix 缓冲退役
            step += 1
        }
        throw M5EngineError.badModel("此段超过 40 秒生成上限，请缩短文本后重试")
    }

    // MARK: 流式解码器（快照恢复 + 首块1帧 + 8帧块 + 尾修剪）

    final class StreamingDecoder {
        unowned let engine: M5Engine
        let wav: WavWriter
        weak let sink: AudioSink?
        let start: Date
        var pending = [Float]()
        var bufferedCodes = [Int64]()
        var chunkTarget = M5Engine.firstChunkFrames
        var past: [Float] = []
        var pastFrames = 0
        var convs: [Float]
        var codecSeconds = 0.0, sinkSeconds = 0.0
        var firstChunkSeconds = -1.0
        let snapshotKey: String

        init(engine: M5Engine, voice: M5Voice, wav: WavWriter, sink: AudioSink?, start: Date) {
            self.engine = engine; self.wav = wav; self.sink = sink; self.start = start
            self.convs = engine.decoderSnapshots[Self.key(voice)] ?? [Float](repeating: 0, count: engine.runner.codecStateBytes / 4)
            self.snapshotKey = Self.key(voice)
            self.pastFrames = engine.snapshotFrames[snapshotKey] ?? 0
        }
        static func key(_ v: M5Voice) -> String { M5Engine.voiceKey(v) }

        func push(_ codePtr: UnsafePointer<Int64>) throws {
            var frame = [Int64](repeating: 0, count: 16)
            for b in 0..<16 { frame[b] = max(0, min(2047, codePtr[b] - 1)) }
            bufferedCodes.append(contentsOf: frame)
            if bufferedCodes.count / 16 >= chunkTarget { try flush() }
        }

        private func flush() throws {
            guard !bufferedCodes.isEmpty else { return }
            let n = bufferedCodes.count / 16
            // 布局转置为 [16, n]
            var transposed = [Int64](repeating: 0, count: bufferedCodes.count)
            for b in 0..<16 { for f in 0..<n { transposed[b * n + f] = bufferedCodes[b + f * 16] } }
            var audio = [Float](repeating: 0, count: n * 1920 + 64)
            var present = [Float](repeating: 0, count: 8 * 2 * 16 * 64 * (pastFrames + n + 8))
            let t0 = Date()
            let (audioCount, _) = try transposed.withUnsafeBufferPointer { tc in
                try past.withUnsafeBufferPointer { tp in
                    try convs.withUnsafeBufferPointer { cv in
                        try audio.withUnsafeMutableBufferPointer { au in
                            try present.withUnsafeMutableBufferPointer { pr in
                                try engine.runner.runCodec(codes: tc.baseAddress!, frames: n, offset: pastFrames,
                                                           past: past.isEmpty ? nil : tp.baseAddress, pastFrames: pastFrames,
                                                           convs: cv.baseAddress, outAudio: au.baseAddress!, outPresent: pr.baseAddress!,
                                                           outConvs: convs.withUnsafeMutableBufferPointer { $0.baseAddress! })
                            }
                        }
                    }
                }
            }
            codecSeconds += Date().timeIntervalSince(t0)
            bufferedCodes.removeAll(keepingCapacity: true)
            let presentLen = 8 * 2 * 16 * 64 * (pastFrames + n)
            past = Array(present[0..<min(presentLen, present.count)])
            pastFrames += n
            pending.append(contentsOf: audio[0..<audioCount])
            try emit()
        }

        private func emit() throws {
            if firstChunkSeconds < 0 {
                // 首块只有 1 帧（1920 样本）时不够 2400 保留阈值：整块直发（与 Android 相同）
                if pending.count >= 1920 {
                    firstChunkSeconds = Date().timeIntervalSince(start)
                    try write(Array(pending))
                    pending.removeAll(keepingCapacity: true)
                    chunkTarget = M5Engine.chunkFrames
                }
                return
            }
            guard pending.count > M5Engine.tailSamples else { return }
            let emit = Array(pending[0..<(pending.count - M5Engine.tailSamples)])
            try write(emit)
            pending.removeFirst(emit.count)
        }

        private func write(_ samples: [Float]) throws {
            let t0 = Date()
            if let sink = sink {
                samples.withUnsafeBufferPointer { sink.write($0.baseAddress!, count: samples.count) }
                sinkSeconds += Date().timeIntervalSince(t0)
            }
            try samples.withUnsafeBufferPointer { try wav.append($0.baseAddress!, count: samples.count) }
        }

        func finish() throws {
            try flush()
            if firstChunkSeconds < 0 {
                if pending.count < M5Engine.tailSamples { throw M5EngineError.badModel("声音解码器返回的长度不足") }
            } else if pending.count < M5Engine.tailSamples {
                throw M5EngineError.badModel("声音解码器返回的长度不足")
            }
            // 丢弃段尾 100ms
            if firstChunkSeconds >= 0 {
                let keep = pending.count - M5Engine.tailSamples
                if keep > 0 { try write(Array(pending[0..<keep])) }
            }
            pending.removeAll()
        }
    }
}

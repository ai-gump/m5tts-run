#!/usr/bin/env python3
"""M5-TTS CPU 后端（路线 2：官方 ONNX + Python），SSE 流式输出，供测试页调用。

- 启动时加载 M5Onnx 一次（4 个 ONNX 图 + embedding），之后每请求只做生成；
- 接口与 audiocpp_server 的 /v1/audio/speech 对齐：pcm + sse（speech.audio.delta / speech.audio.done）；
- 参考编码按 sha1 落盘缓存（--ref-cache 机制），跨请求复用；
- 端口默认 8092。单请求串行（M5Onnx 是单请求 runtime）。
"""
import base64
import hashlib
import io
import json
import os
import sys
import tempfile
import threading
import time
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 仓库根（web/ 的上一级）
M5_DIR = os.path.join(ROOT, "M5-TTS")
sys.path.insert(0, M5_DIR)

from infer_onnx import M5Onnx  # noqa: E402

WEIGHTS = os.path.join(ROOT, "weights")
DEFAULT_REF = os.path.join(M5_DIR, "assets", "aslp_default_voice.wav")
DEFAULT_REF_TEXT = "猪携带该疾病，然后通过蚊子传染给人类。"
REF_CACHE_DIR = os.path.join(ROOT, "outputs", "refcache")
TMP_DIR = os.path.join(ROOT, "outputs", "tmp_refs")
os.makedirs(REF_CACHE_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)

print("loading M5Onnx (CPU)...", file=sys.stderr)
MODEL = M5Onnx(model=WEIGHTS, threads=8, local_files_only=True)
print("model ready", file=sys.stderr)

_LOCK = threading.Lock()


def ref_cache_path(wav_path: str) -> str:
    h = hashlib.sha1(open(wav_path, "rb").read()).hexdigest()
    return os.path.join(REF_CACHE_DIR, f"ref_{h}.npz")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def do_POST(self):
        if self.path != "/v1/audio/speech":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(length) or b"{}")
        text = (req.get("input") or "").strip()
        if not text:
            self.send_error(400, "input required")
            return

        prompt_wav, prompt_text = DEFAULT_REF, DEFAULT_REF_TEXT
        ref = req.get("voice_ref")
        if isinstance(ref, dict) and ref.get("type") == "base64" and ref.get("data"):
            raw = base64.b64decode(ref["data"].split(",")[-1])
            fd, tmp = tempfile.mkstemp(suffix=".wav", dir=TMP_DIR)
            os.close(fd)
            with open(tmp, "wb") as f:
                f.write(raw)
            prompt_wav = tmp
            prompt_text = (req.get("reference_text") or "").strip() or DEFAULT_REF_TEXT
        elif (req.get("reference_text") or "").strip():
            prompt_text = req["reference_text"].strip()

        first_frames = int(req.get("first_frames") or 2)
        left_context = int(req.get("left_context") or 4)
        seed = int(req.get("seed") or 1234)
        ref_cache = ref_cache_path(prompt_wav)
        ref_cache_hit = os.path.exists(ref_cache)

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        started = time.perf_counter()
        first_pcm = None
        emitted = 0
        chunks = 0
        try:
            with _LOCK:
                for pcm in MODEL.generate(
                    text, prompt_wav, prompt_text,
                    stream=True, first_frames=first_frames, chunk_frames=4,
                    left_context=left_context, seed=seed, max_frames=500,
                    ref_cache=ref_cache,
                ):
                    if first_pcm is None:
                        first_pcm = time.perf_counter() - started
                    # 页面按 s16le 解码；转成 s16 保证与 audiocpp 输出一致
                    s16 = (np.clip(pcm, -1, 1) * 32767).astype("<i2").tobytes()
                    b64 = base64.b64encode(s16).decode()
                    self.wfile.write(
                        f"data: {json.dumps({'type': 'speech.audio.delta', 'audio': b64})}\n\n".encode())
                    self.wfile.flush()
                    emitted += len(pcm)
                    chunks += 1
            gen_ms = (time.perf_counter() - started) * 1000
            audio_ms = emitted / 24000 * 1000
            rtf = gen_ms / audio_ms if audio_ms > 0 else 0.0
            print(
                f"[CPU_PERF] ttft_ms={(first_pcm or 0) * 1000:.1f} gen_ms={gen_ms:.1f} "
                f"audio_ms={audio_ms:.1f} rtf={rtf:.3f} realtime={(audio_ms / gen_ms if gen_ms > 0 else 0):.2f} "
                f"chunks={chunks} chars={len(text)} ref_cache={'hit' if ref_cache_hit else 'miss'}",
                file=sys.stderr, flush=True)
            timing = {"ttft_ms": round((first_pcm or 0) * 1000, 1),
                      "audio_ms": round(audio_ms, 1)}
            self.wfile.write(f"data: {json.dumps({'type': 'speech.audio.done', 'timing': timing})}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except BrokenPipeError:
            pass


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8092"))
    print(f"cpu server on http://127.0.0.1:{port}/v1/audio/speech", file=sys.stderr)
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()

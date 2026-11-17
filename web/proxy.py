#!/usr/bin/env python3
"""M5-TTS 流式播放测试台：静态页 + SSE 反代（同源，避免 CORS）。

用法：python3 web/proxy.py  → 浏览器打开 http://127.0.0.1:8091
上游：audiocpp_server @ http://127.0.0.1:8090（/v1/audio/speech，SSE + pcm）
"""
import http.server
import socketserver
import urllib.request
import urllib.error
import os
import json

PAGE_DIR = os.path.dirname(os.path.abspath(__file__))
UPSTREAM_GPU = "http://127.0.0.1:8090/v1/audio/speech"   # audiocpp_server (Metal)
UPSTREAM_CPU = "http://127.0.0.1:8092/v1/audio/speech"   # web/cpu_server.py (ONNX, infer_onnx)


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=PAGE_DIR, **kwargs)

    def log_message(self, fmt, *args):  # 安静一点
        pass

    def do_POST(self):
        from urllib.parse import urlparse, parse_qs
        parsed = urlparse(self.path)
        if parsed.path != "/api":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        req = urllib.request.Request(
            (UPSTREAM_CPU if parse_qs(parsed.query).get("backend", ["gpu"])[0] == "cpu" else UPSTREAM_GPU),
            data=body, method="POST",
            headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        )
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("X-Accel-Buffering", "no")
                self.end_headers()
                while True:
                    chunk = resp.read1(8192)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")
            self.send_response(e.code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": detail}).encode())


class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8091"))
    print(f"test bench on http://127.0.0.1:{port}  (gpu {UPSTREAM_GPU} | cpu {UPSTREAM_CPU})")
    Server(("127.0.0.1", port), Handler).serve_forever()

# M5-TTS 流式播放测试台（Web）

路线 1（audio.cpp · Metal）的流式体验测试：网页输入文本 → SSE 接收 PCM → Web Audio 实时播放。
两个测试点：**首帧延迟**（点击 → 第一块音频到达）与**流畅度**（按 1× 虚拟播放时钟判定欠载/卡顿）。

## 启动（四件套）

页面支持 **GPU（Metal · audio.cpp）与 CPU（ONNX · infer_onnx）双后端**，顶部 radio 切换。

```bash
# 1) GPU 后端：audiocpp_server（Metal，配置里已挂 m5_tts streaming + 默认参考音色）
cd ~/Documents/Project/m5tts-run
./audio.cpp/build/bin/audiocpp_server --config server-config.json --backend metal --no-ui &

# 2) CPU 后端：web/cpu_server.py（常驻加载 M5Onnx，SSE 输出，端口 8092；加载约 15s）
~/Documents/Project/m5tts-run/.venv/bin/python web/cpu_server.py &

# 3) 同源代理（页面 + /api 转发，按 ?backend=gpu|cpu 路由，避免 CORS）
python3 web/proxy.py &

# 4) 浏览器打开
open http://127.0.0.1:8091/tts-test.html
```

CPU 后端包含三项首帧优化（`--ref-cache` 参考编码缓存 / `--first-frames 2` / `--left-context 4`），实测首帧 ~0.2s；页面会按后端自动调整播放启动缓冲（CPU 300ms / GPU 150ms，CPU 生成余量小）。

先点「预热模型」（首次请求会懒加载 1GB 权重 + 图预热），再输入文本点「生成并流式播放」。

## 页面指标

| 指标 | 含义 |
|---|---|
| 首帧延迟（页面实测） | 点击 → 第一个 PCM 块到达的时间 |
| 首帧延迟（服务端 ttft） | `speech.audio.done` 的 `timing.ttft_ms`，用于交叉核对 |
| 播放欠载（卡顿） | 按 1× 虚拟播放时钟：数据到达晚于播放头 = 欠载，记录次数和累计 ms |
| 最小缓冲余量 | 播放中「已调度音频领先播放头」的最小值，越小越惊险 |
| 生成实时倍数 | 音频时长 ÷ 生成耗时，>1 表示生成喂得饱播放 |
| 音频时长 / 块数 | PCM s16le 24 kHz |

## 服务端性能日志（与页面六卡对应）

不用打开浏览器也能核对：两个后端每请求都会往 stderr 打一行汇总。

| 后端 | 日志行 | 字段 |
|---|---|---|
| GPU（audiocpp_server） | `[M5_PERF] ...` | `frames` / `audio_ms` / `wall_ms` / `rtf`（wall÷audio）/ `realtime`（audio÷wall）/ `ttfa_ms` |
| CPU（cpu_server.py） | `[CPU_PERF] ...` | `ttft_ms` / `gen_ms` / `audio_ms` / `rtf` / `realtime` / `chunks` / `chars` / `ref_cache=hit\|miss` |

启动时把 stderr 一并落盘即可长期观察：

```bash
./audio.cpp/build/bin/audiocpp_server --config server-config.json --backend metal --no-ui 2>&1 | tee gpu-server.log
python web/cpu_server.py 2>&1 | tee cpu-server.log
```

`ref_cache` 用来验证参考编码缓存：同一参考音频首次 `miss`、之后 `hit`，首帧应从 ~600ms 降到 ~100ms 量级。

**对比提醒**：页面「生成实时倍数」从点击算到流结束（含首帧与网络），服务端 `rtf`/`realtime` 是单请求生成窗口，数值会略有差异；与课程实测表对比时以服务端日志为准。

### 参考音色：默认 or 上传

页面提供两种参考音色模式：

- **默认音色**：使用 `server-config.json` 里配置的默认参考（`M5-TTS/assets/aslp_default_voice.wav`）；
- **上传参考音频**：选择本地音频（任意浏览器可解码格式，页面内统一转为 **24kHz 单声道 WAV** 后以 base64 inline 发送，server 限制 base64 ≤5 MiB），并**必须填写与音频逐字一致的转写文字**（M5 克隆要求音+字成对）。

实测（上传 8.1s 慢速参考，目标文本 16 字）：第一次请求 630ms（新参考编码 + 状态 prime + 建快照），第二次 129ms、第三次 89ms（参考编码缓存 + 快照恢复），全程 0 欠载；输出 4.30s 音频——**语速跟随参考音频（3.7 字/秒）**，默认参考则是 3.10s（5.2 字/秒）。

### 已知局限（M3 实测，2026-11-17）

1. **语速偏快、且没有原生语速参数**：模型请求选项只有 language/temperature/top_k/max_tokens/seed。但实测发现**语速会跟随参考音频迁移**——默认参考输出 5.2 字/秒（16 字 / 3.10s），换一段 3.7 字/秒的慢速参考后，同文本输出也降到 3.7 字/秒（4.30s）。**首选解法：换慢速参考音频（保音调）**；页面滑杆的 0.75–1.5× 是播放端兜底（会轻微改音调）。
2. **自然度偏平、缺少起伏**：与论文 10 人听测结论一致（自然度略低于 MOSS-TTS-Nano）。原因：50M 容量 + 24 kHz 上限；GRPO 后训练优化的是可懂度/相似度/停顿，不优化表现力；论文 roadmap 也承认没有情绪、语气、笑声叹气的显式控制。缓解：temperature 可略调高、用更有表现力的参考音频；对韵律要求高的场景建议对比 MOSS-TTS-Nano 或更大模型。

### 语速

页面提供 **0.75×–1.5× 语速滑杆**。M5-TTS 模型本身不提供语速参数（请求选项里只有 language/temperature/top_k/max_tokens/seed），因此这里是**播放端变速**（Web Audio `playbackRate`）：优点是零延迟、不重新生成；代价是"磁带变速"对音调有轻微影响（0.85×–1.2× 区间不明显）。虚拟播放时钟会按速率换算，欠载检测在变速下依然有效——0.85× 与 1.2× 实测均 0 欠载（见上表同年实测）。

## 实测参考（M3 / 8 线程 / Metal，2026-11-17）

| 后端 | 场景 | 首帧（页面/服务端） | 欠载 | 实时倍数 | 最小缓冲 |
|---|---|---|---|---|---|
| **GPU**（Metal） | 短句 3.10s | 82～110 / 79～107 ms | **0 次** | ~2.0× | ~205 ms |
| **GPU**（Metal） | 80 字长文 18.86s | 74 / 73 ms | **0 次** | 2.26× | 209 ms |
| **CPU**（ONNX） | 短句 2.86s | 215～220 / 212～217 ms | **0 次** | ~1.4× | 360 ms |
| **CPU**（ONNX） | 80 字长文 19.58s | 218 / 215 ms | **0 次** | 1.49× | 360 ms |

（各场景 2 轮；GPU 缓冲 150ms、CPU 缓冲 300ms。GPU 含解码器状态快照；CPU 含参考编码缓存 + 首块 2 帧 + left_context 4。）

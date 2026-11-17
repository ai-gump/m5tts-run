# W82 墨水屏平板实测记录

> 设备专属数据，**不要与华为 P40 Pro 的数据混在一起**。W82 是低功耗墨水屏设备，CPU/GPU/内存与手机完全不同。

## 设备与测试条件

| 项 | 值 |
|---|---|
| 设备 | W82（wetao，墨水屏平板） |
| SoC | Rockchip RK3566（4×Cortex-A55，`ro.board.platform=rk356x`） |
| 内存 | 4GB（MemTotal ≈ 3.93GB） |
| 系统 | Android 11（`ro.build.version.sdk=30`） |
| ABI | arm64-v8a |
| 存储 | 51GB 可用 |
| 同模型 | 与 P40 Pro 相同的 M5-TTS 50M FP32 ONNX 外置模型（1.1GB） |
| 状态 | APK 已安装，模型已复制到应用内部目录，已完成两组生成 |

## 实测数据

测试文本与 P40 Pro 相同：「你好，这是在手机上运行的离线语音合成测试。」（21 字），4 线程，AudioTrack 缓冲 2000ms。当前 APK 包含：解码器状态快照、音色预热、首块 1 帧、后续 8 帧解码块、CPU 占用统计。

| 轮次 | 模型加载 | 音色预热 | 首块延迟 | 生成耗时 | 音频 | RTF | 实时倍数 | 峰值 PSS | CPU 均 | 欠载 |
|---|---|---|---|---|---|---|---|---|---|---|
| 冷启 | 23.1 s | 16.7 s | 2349 ms | 45.77 s | 4.06 s | 11.274 | 0.09× | 1434 MiB | 436% | 1 次 |
| 热态 | 0 | 0 | **2169 ms** | 45.13 s | 4.06 s | **11.115** | 0.09× | 1455 MiB | 273% | 2 次 |

热态分段探针：

```
[M5_PRIME] prime_ms=3 ref_frames=43 source=snapshot
[M5_TOKENIZE] tokenize_ms=7.4
[M5_PREFILL] prefill_ms=1243.0 positions=86
[M5_FIRST_FRAME] first_frame_ms=1612.7
[M5_DEC_FIRST] decode_ms=547.7 frames=1
[M5_SEG] gen_ms=45118 llm_ms=24362 codec_ms=20756 rtf=11.113
[M5_TTFA] first_chunk_ms=2169
[M5_PERF] device=W82 lang=Chinese threads=4 chars=21 segments=1 audio_ms=4060 gen_ms=45126 wall_ms=46186 rtf=11.115 realtime=0.09 first_chunk_ms=2169 load_ms=0 prepare_ms=0 peak_pss_mib=1455 underruns=2 buffer_ms=2000 cpu_pct=273 streaming=true tn=true
```

## 结论

1. **W82 不具备实时合成能力**：RTF 11.1，生成 4.06s 音频需要 45s；LLM 24.4s + codec 20.8s，CPU 已跑满 4 个 A55 核（436% 冷启、273% 热态）。
2. **TTFA 2.17s**：prime 已被快照消掉（3ms）；首帧累计 1.61s（其中 prefill 1.24s，首帧日志从 tokenize 起算）+ 首块解码 0.55s ≈ 2.17s。这是 RK3566 的算力下限，不是 prime 问题。
3. **与 P40 Pro 不可比**：同一份 APK/模型/文本，P40 Pro 热态 TTFA 427ms、RTF 2.04；W82 分别慢 5.1× / 5.4×。墨水屏设备的定位是低功耗阅读，不是实时语音。
4. 要在 W82 上接近实时，需要数量级手段（INT8 量化、NPU 委托、换更小模型或云端合成），仅靠线程/分块调优不可能从 11× 追到 1×；300ms TTFA 在这台硬件上不现实。

## 注意

- 墨水屏刷新率低，UI 指标更新会滞后，性能判断以 `adb logcat -s M5Perf` 为准；
- 4GB 内存下峰值 PSS 约 1.45GB，未触发系统杀进程，但余量不大；
- 冷启模型加载 23s、音色预热 17s，用户等待墙钟远大于 TTFA；性能面板会单列这两项，不要与 TTFA 混算。

# 华为 P40 Pro（ELS-AN00）实测记录

> 设备专属数据，**不要与 W82 墨水屏平板的数据混在一起**。两份数据的 SoC、内存、系统都不同。

## 设备与测试条件

| 项 | 值 |
|---|---|
| 设备 | Huawei ELS-AN00（P40 Pro） |
| SoC | Kirin 990 5G（4×A76 + 4×A55） |
| 内存 | 8GB |
| 系统 | Android 10 / EMUI（`ro.build.version.sdk` 29） |
| ABI | arm64-v8a |
| 模型 | M5-TTS 50M FP32 ONNX，Android 外置模型目录 1.1GB |
| 线程 | 默认 4 线程（另测 8 线程） |
| 文本 | 「你好，这是在手机上运行的离线语音合成测试。」（21 字） |
| 参考音色 | 43 帧 codec codes（默认参考音频） |
| 音频输出 | 24kHz 单声道，AudioTrack 流式，缓冲 2000ms |
| 测量 | App 内 `M5Perf` 日志 + 性能面板；TTFA 从 `synthesize()` 开始到首块 PCM 就绪 |

口径：`gen_ms` = 生成/解码耗时，**不含** AudioTrack 写入阻塞；`wall_ms` = 请求总墙钟；RTF = `gen_ms / audio_ms`，越小越快。

## 优化阶段汇总

### 1. 基线（原始 Demo 逻辑，未优化）

| 轮次 | 线程 | 首块延迟 | 生成耗时 | 音频 | RTF | 实时倍数 | 峰值 PSS | 欠载 |
|---|---|---|---|---|---|---|---|---|
| 冷启（含模型加载） | 4 | 4123 ms | 11.70 s | 4.06 s | 2.881 | 0.35× | 1282 MiB | 未打点 |
| 热态 | 4 | 3744 ms | 11.21 s | 4.06 s | 2.761 | 0.36× | 1361 MiB | 未打点 |
| 热态（欠载打点版） | 4 | 4031 ms | 11.91 s | 4.06 s | 2.934 | 0.34× | 1253 MiB | 3 次 |
| 热态 | 8 | 3563 ms | 11.49 s | 4.06 s | 2.830 | 0.35× | 1285 MiB | 3 次 |

基线阶段的分段探针（4 线程，热态）：

```
[M5_PRIME] prime_ms=3067 ref_frames=43
[M5_PREFILL] prefill_ms=78.1 positions=86
[M5_FIRST_FRAME] first_frame_ms=157.8
[M5_DEC_FIRST] decode_ms=277.3 frames=4
[M5_SEG] gen_ms=11519 llm_ms=7671 codec_ms=3848 rtf=2.837
[M5_TTFA] first_chunk_ms=3784
[M5_PERF] rtf=2.838 realtime=0.35 first_chunk_ms=3784 load_ms=4651 peak_pss_mib=1271 underruns=3
```

结论：**TTFA 的 3.78s 里 3.07s 是参考音色 prime**（43 帧解码器状态预热），首帧/首块计算只占 ~0.5s；生成阶段 LLM 7.67s、codec 3.85s。

### 2. 解码器快照 + 音色预热 + 首块 2 帧 + 8 帧解码块

| 轮次 | 首块延迟 | 生成耗时 | 音频 | RTF | 实时倍数 | 峰值 PSS | 欠载 |
|---|---|---|---|---|---|---|---|
| 冷启（模型加载 4.79s + 音色预热 3.14s） | 710 ms | 8.58 s | 4.06 s | 2.114 | 0.47× | 1485 MiB | 0 次 |
| 热态（引擎与快照已在内存） | **427 ms** | 8.29 s | 4.06 s | **2.042** | 0.49× | 1475 MiB | 0 次 |

热态分段探针：

```
[M5_PRIME] prime_ms=1 ref_frames=43 source=snapshot
[M5_TOKENIZE] tokenize_ms=0.7
[M5_PREFILL] prefill_ms=76.2 positions=86
[M5_FIRST_FRAME] first_frame_ms=152.6
[M5_DEC_FIRST] decode_ms=155.1 frames=2
[M5_SEG] gen_ms=8289 llm_ms=4567 codec_ms=3720 rtf=2.042
[M5_TTFA] first_chunk_ms=427
[M5_PERF] rtf=2.042 realtime=0.49 first_chunk_ms=427 load_ms=0 prepare_ms=0 peak_pss_mib=1475 underruns=0
```

结论：

- **TTFA 从 ~3.8s 降到 427ms**（-89%）：prime 3.07s → 1ms（快照恢复），首块 4 帧 → 2 帧（277ms → 155ms）；首帧日志 153ms 已包含 prefill 76ms，不要再相加；
- **RTF 从 ~2.84 降到 2.04**：首块 2 帧 + 后续 8 帧一块减少了解码调用；LLM 7.67s → 4.57s（含首次/热态差异，见备注），codec 3.85s → 3.72s；
- **仍未达到实时**（RTF > 1）：生成 4.06s 音频需要 8.29s；欠载计数虽然为 0，但按 RTF 推断播放端无法持续供数，**不能据此认为播放流畅**；
- 峰值 PSS 从 ~1.3GB 升到 ~1.5GB（快照/ONNX arena 波动），仍低于 8GB 设备的可用内存。

### 3. 已编译、未上机验证的下一版（不要引用为实测）

- 首块改 1 帧（1920 样本，绕开 2400 样本尾修剪阈值），预期 TTFA 再降约 78ms；
- UI/日志增加 CPU 平均占用率（`Process.getElapsedCpuTime()`）；
- XNNPACK 实验：一次冷启首帧 733ms（原 152ms），未完成热态复测，已回退，**不作为结论**。

## 备注与未决项

1. 基线阶段首轮 TTFA 包含 `synthesize()` 内的 3.07s prime；模型加载（4.7s）和音色预热（3.1s）**不计入 TTFA**，但计入用户等待墙钟，性能面板会单列；
2. 热态 LLM 从 7.67s 降到 4.57s 不是单变量结果：这一轮同时改了首块/解码块，且前一轮的 JIT/频率状态不同，后续需要在同一版本上做 A/B 复测；
3. 欠载计数来自 `AudioTrack.getUnderrunCount()`，RTF>1 时显示 0 次，和理论预期矛盾，需用真实听感或播放头/写入量交叉验证；
4. 8 线程在基线上只把 RTF 从 2.93 降到 2.83、TTFA 从 4031ms 降到 3563ms，收益远小于 4→8 的核数增加，符合“小矩阵喂不饱多核”的判断。

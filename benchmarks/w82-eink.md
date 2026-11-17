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

## FP32 vs FP16 对比（ORT 1.32.0，2026-11-17 16:2x）

先说明一个**运行时前提**：FP16 模型（`keep_io_types=True` 转换）在 ORT Android **1.23.2 上无法加载**，报 `GraphUtils GetIndexFromName ... InsertedPrecisionFreeCast_... does not exist`；把依赖升到 **onnxruntime-android 1.32.0**（APK 188MB→223MB）后才能加载。因此本节 A/B 全部基于 ORT 1.32.0；上一节的数据是 1.23.2，**不要跨运行时版本直接相减**。FP32 在 1.23.2 与 1.32.0 下性能基本持平（热态 RTF 11.1 vs 11.0）。

同一台 W82、同一 APK、同一段 21 字文本、4 线程；FP16 = 三张 ONNX 转 FP16（I/O 保持 FP32），embedding 仍 FP32，模型目录 1.1GB→734MB：

| 指标 | FP32 冷启 | FP32 热态 | FP16 冷启 | FP16 热态 |
|---|---|---|---|---|
| 模型加载 | 24.9 s | — | 27.5 s | — |
| 音色预热 | 15.9 s | — | 23.6 s | — |
| 首块延迟 | 2523 ms | 2295 ms | 3884 ms | **3286 ms** |
| 生成耗时 | 45.19 s | 44.52 s | 72.18 s | 70.28 s |
| 音频时长 | 4.06 s | 4.06 s | 4.30 s | 4.30 s |
| 生成帧数 | 52 | 52 | 55 | 55 |
| RTF | 11.130 | 10.965 | 16.785 | **16.343** |
| 其中 LLM | 24.66 s | 24.41 s | 39.54 s | 38.74 s |
| 其中 codec | 20.52 s | 20.11 s | 32.63 s | 31.53 s |
| 峰值 PSS | 1449 MiB | 1470 MiB | **984 MiB** | **983 MiB** |
| CPU 均 | 437% | 275% | 450% | 310% |
| 欠载 | 1 | 2 | 1 | 1 |

热态对比结论：

- **速度：FP16 全面更慢**——TTFA +43%（2295→3286ms）、RTF +49%（10.97→16.34）、LLM +59%、codec +57%。与 Mac CPU 上的方向一致：ORT CPU EP 的 FP16 kernel 覆盖有限，大量节点要 Cast 回 FP32；
- **资源：FP16 省三分之一**——模型目录 1.1GB→734MB，峰值 PSS 1470→983MiB（-33%）。对 4GB 内存设备，这是 FP16 唯一站得住的理由；
- **质量：待听感验证**。FP16 多生成 3 帧（音频 4.30s vs 4.06s）；波形相关性在同设备 FP32 vs FP16 只有 0.088，但同精度跨设备（Mac FP32 vs W82 FP32）相关性也≈0，说明采样路径对数值扰动极敏感，**波形相关不能当质量结论**，需要人工听可懂度/音色相似度；
- **W82 最终建议：保持 FP32**。FP16 换不来速度，只省内存/磁盘；除非存储或内存告急，否则不换。

## INT8 动态量化（depth+codec，ORT 1.32.0，2026-11-17 18:1x）

组合选择在 Mac 上先做过（见 `model-formats.md`）：全量 INT8 / 仅 backbone 都会破坏 EOS（不停机或时长翻倍），**可用组合是 depth(MTP)+codec 量化、backbone 保 FP32**。Android 包 694MB（depth 68MB + codec 195MB + backbone/Embedding 保 FP32）。

| 指标 | INT8 冷启 | INT8 热态 | （对照）FP32 热态 |
|---|---|---|---|
| 模型加载 | 16.5 s | — | 24.9 s |
| 音色预热 | 10.0 s | — | 15.9 s |
| 首块延迟 | **1044 ms** | **1138 ms** | 2295 ms |
| 生成耗时 | 38.97 s | 38.50 s | 44.52 s |
| 音频时长 / 帧数 | 6.22 s / 79 帧 | 6.22 s / 79 帧 | 4.06 s / 52 帧 |
| RTF | 6.27 | **6.19** | 10.97 |
| 其中 LLM | 20.91 s | 20.68 s | 24.41 s |
| 其中 codec | 18.06 s | 17.81 s | 20.11 s |
| 峰值 PSS | 914 MiB | 914 MiB | 1470 MiB |
| CPU 均 | 344% | 230% | 275% |
| 欠载 | 2 | 2 | 2 |

热态三方对比（同一 APK、同一段 21 字文本）：

| | FP32 | FP16 | **INT8（depth+codec）** |
|---|---|---|---|
| 首块延迟 | 2295 ms | 3286 ms | **1138 ms（-50%）** |
| RTF | 10.97 | 16.34 | **6.19（提速 1.77×）** |
| 单帧成本 | 856 ms/帧 | — | 487 ms/帧（提速 1.76×，与 Mac 一致） |
| 峰值 PSS | 1470 MiB | 983 MiB | **914 MiB** |
| 模型目录 | 1.1GB | 734MB | **694MB** |
| CPU 均 | 275% | 310% | 230% |

**质量警告（必须人工听感后再定采用）**：同一段文本，FP32 生成 52 帧/4.06s，INT8 在 W82 上生成 **79 帧/6.22s**（拖长约 50%）；能量全程有语音、无长静音（每秒 RMS 0.038～0.075），说明不是静音尾巴，但无法从数字判断是“语速变慢”还是“复读/多说内容”。同一组合在 Mac 上是 53 帧/4.14s（与 FP32 一致）——**同权重跨设备轨迹差异大**（此前同精度跨设备波形相关也≈0）。样本已存：`benchmarks/samples/w82-fp32.wav`、`benchmarks/samples/w82-int8-depth-codec.wav`，请人耳对比后再决定是否把 INT8 设为默认。

设备目录现状：`files/m5-model`（FP32，默认）、`files/m5-model-fp16`、`files/m5-model-int8`，用 `run-as mv` 切换后 force-stop 重启即可。

## 注意

- 墨水屏刷新率低，UI 指标更新会滞后，性能判断以 `adb logcat -s M5Perf` 为准；
- 4GB 内存下峰值 PSS 约 1.45GB，未触发系统杀进程，但余量不大；
- 冷启模型加载 23s、音色预热 17s，用户等待墙钟远大于 TTFA；性能面板会单列这两项，不要与 TTFA 混算；
- 设备上同时保留 `files/m5-model`（FP32，当前默认）与 `files/m5-model-fp16`（734MB），切换方式：`adb shell run-as com.m5.offline mv files/m5-model files/m5-model-fp32 && adb shell run-as com.m5.offline mv files/m5-model-fp16 files/m5-model`，切换后 force-stop 再启动。

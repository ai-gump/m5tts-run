# M5-TTS 权重大小与精度形态

> 三个数字经常被混着说：**50M 参数**、**1.77GB ONNX 包**、**1.14GB Android 子集**。这里把口径拆开，并记录本次 FP16 实验的实际体积。

## 一张表

| 形态 | 包含内容 | 大小 | 精度 |
|---|---|---|---|
| 官方 HF ONNX 全量包 | `weights/` 12 个文件 | 1.77GB | FP32 |
| Android 实际运行子集 | backbone 116MB + depth 267MB + codec_decoder_cached 435MB + text_embeddings 254MB + audio_embeddings 64MB + tokenizer 5.2MB | 约 1.14GB | FP32 |
| audio.cpp GGUF | `m5-voiceclone-f16.gguf` | 954MB | LM 权重 F16，codec 仍 F32 |
| **本次 FP16 ONNX（模型 FP16，embedding 仍 FP32）** | backbone 58MB + depth 134MB + codec 218MB + embeddings 318MB + tokenizer 5.2MB | **约 733MiB（753MB）** | 模型 FP16，I/O 保持 FP32；embedding FP32 |
| **全 FP16 目标形态（embedding 也转 FP16，未实现）** | 模型 430MB + embedding 159MB + tokenizer 5.2MB | 约 594MB | 全 FP16 |

## 为什么不是“50M 参数 = 50MB”

- **50M 只是语言模型部分**：30M 时序骨干 + 20M 帧内 MTP 解码器；不含 codec、不含 embedding；
- 文本 embedding：129,992 × 512 × 4B ≈ 254MB；
- 音频 embedding：16 × 2051 × 512 × 4B ≈ 64MB；
- codec decoder：约 435MB（FP32）；
- GGUF 文件名里的 `f16` 只表示 **LM 权重 F16**，官方打包说明明确保留 F32 codec 权重（GGUF 元数据：“published native packages retain F32 codec weights”）。

## 本次 FP16 ONNX 的做法

- 用 `onnxruntime.transformers.float16.convert_float_to_float16(model, keep_io_types=True, disable_shape_infer=True)` 转换 `backbone.onnx`、`depth_frame.onnx`、`codec_decoder_cached.onnx`；
- `keep_io_types=True` 让模型输入/输出仍是 FP32，Android Java 侧不用改 `FloatBuffer`/`OnnxTensor` 的喂法；图内部插 Cast 转 FP16；
- embedding 仍用 FP32，第一轮对比先隔离“ONNX 模型精度”这一个变量；
- 转换后体积：backbone 121.8→61.0MB、depth 280.0→140.6MB、codec 456.5→228.3MB；三张图合计约 430MB；
- 转换告警：`Sin`/`Cos`/`Exp` 等节点在 CPU EP 没有 FP16 kernel，ORT 会在运行时插 Cast 或回退；这正是 FP16 在 CPU 上不一定更快的原因。

## Mac 全流程对照（ORT 1.32 CPU / 4 线程，仅作方向参考）

同一份参考音频、同一段 21 字文本、`--stream --first-frames 1 --left-context 4`：

| 指标 | FP32 | FP16（模型 FP16，I/O FP32） | 变化 |
|---|---|---|---|
| ONNX 模型体积 | 856MB | 430MB | -50% |
| 模型目录总体积 | 1.14GB | 733MiB | -36% |
| 音频时长 | 4.14s | 4.22s | +2% |
| 生成耗时 | 3.13s | 4.44s | **+42%** |
| RTF | 0.755 | 1.051 | **变慢** |
| 首块延迟 | 681ms | 832ms | **+22%** |
| 音频相关性 | — | 0.806 | 有可听差异 |

结论：**在 CPU EP 上 FP16 换来的是体积减半，不是速度提升**。原因是 ORT CPU 的 FP16 kernel 覆盖有限，`Sin/Cos/Exp` 等节点需要 Cast 回 FP32；Mac 单算子测试也是同样方向（backbone 0.94×、depth 0.69×、codec 0.96×）。Android 真机数据待设备重连后补；如果 Android 也是这个趋势，FP16 的定位应是“省磁盘/省下载”，而不是“跑得动”。

## 预期与风险（已由真机验证）

- **体积/磁盘**：模型从 1.14GB 降到 753MB，若 embedding 也转 FP16 可到约 594MB；对存储紧张的设备有意义；
- **速度**：ORT CPU EP 的 FP16 支持有限，Mac 全流程 FP16 比 FP32 慢 42%（RTF 0.755→1.051），Android 真机实测 +49%（W82，见下节）；
- **内存**：FP16 权重映射减半，但运行时 Cast/回退可能把激活值又升回 FP32，峰值 PSS 不一定等比例下降；
- **质量**：FP16 动态范围比 FP32 小，转换时已对极端值做截断（`-3.4e38 → -10000`、极小正值 → `1e-7`）；需要用同一文本做音频相关性/听感对比。

## Android 真机 A/B（W82 / RK3566 / ORT 1.32.0，热态）

**运行时前提**：FP16 模型在 ORT Android **1.23.2 加载失败**（`GraphUtils GetIndexFromName ... InsertedPrecisionFreeCast ... does not exist`），升到 **1.32.0** 才能跑（APK 188MB→223MB）。FP32 在两个版本下性能持平。

| 指标（热态） | FP32 | FP16（模型 FP16，I/O FP32） | 变化 |
|---|---|---|---|
| 模型目录 | 1.1GB | 734MB | **-33%** |
| 首块延迟 | 2295 ms | 3286 ms | **+43%** |
| RTF | 10.97 | 16.34 | **+49%** |
| LLM / codec 分段 | 24.4s / 20.1s | 38.7s / 31.5s | +59% / +57% |
| 峰值 PSS | 1470 MiB | 983 MiB | **-33%** |
| CPU 均 | 275% | 310% | +13% |

真机结论与 Mac 一致且更明显：**FP16 在 CPU 上是“省内存/磁盘”选项，不是提速选项**；W82 上建议保持 FP32。质量方面 FP16 多生成 3 帧（4.30s vs 4.06s），采样路径对数值扰动敏感（同精度跨设备波形相关也≈0），需要人工听感验证后才能下质量结论。完整冷/热数据见 `benchmarks/w82-eink.md`。

## 相关文件

- 华为 P40 Pro：`benchmarks/huawei-p40-pro.md`
- W82 墨水屏（含 FP32/FP16 A/B）：`benchmarks/w82-eink.md`

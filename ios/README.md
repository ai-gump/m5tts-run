# M5 离线语音 iOS Demo

对标 `android/` 的 iOS 版：同一套 ONNX FP32 模型（10 个文件）、同样的优化流水线（解码器状态快照 + 音色预热 + 首块 1 帧 + 8 帧块）、同样的性能观测（`[M5_PERF]` 日志 + 面板）。**程序化 UIKit，无 storyboard**；`xcodegen` 生成工程。

## 1. 先拿到 OnnxRuntime xcframework（必做，约 60MB）

`ios/frameworks/` 里没有也不入库框架本体，三条途径任选：

| 途径 | 做法 |
|---|---|
| A. Mac 网络可达 GitHub | `./scripts/download_framework.sh 1.23.2`（脚本内已带 ghproxy 备选） |
| B. 手机热点 | Mac 连手机热点后执行 A |
| C. Linux 侧代下 | 在可达网络的 Linux 机器上下载 `https://github.com/microsoft/onnxruntime/releases/download/v1.23.2/onnxruntime-ios-1.23.2.zip`，拷回后放进 `ios/frameworks/` 再跑一次 A（脚本会直接解压） |

产物：`ios/frameworks/onnxruntime.xcframework`。**本仓库提交的版本要求与 Android 同代（1.23.x/1.32.x 均可，FP16 模型需 ≥1.32）。**

> 注意：本次开发网络对 `microsoft/onnxruntime` 的 GitHub 资源整体返回伪 404（rockchip 等仓库正常），所以在 Mac 上脚本可能失败——这是网络问题，不是链接失效。

## 2. 打开工程并真机安装

```bash
cd ios
open M5Demo.xcodeproj
```

Xcode 里：target **M5Demo → Signing & Capabilities → Team** 选你的开发者账号（工程默认关闭了签名，真机跑需要选 Team，bundle id `com.aigump.m5demo` 冲突可自行改）→ 选真机 → Run。

Info.plist 已开启 `UIFileSharingEnabled` / `LSSupportsOpeningDocumentsInPlace`（访达可见 Documents）。

## 3. 导入模型（与 Android 完全同构）

把 `outputs/android_model_staging/` 的 **10 个文件**（`backbone.onnx`、`depth_frame.onnx`、`codec_decoder_cached.onnx`、`text_embeddings.npy`、`audio_embeddings.npy`、`tokenizer.bin`、`voices.json`、`manifest.json`、`codec_cached.json`、`bundle.json`）放进 App 的 `Documents/m5-model/`：

- **访达**：iPhone 连 Mac → 访达侧栏选设备 → 文件 → M5Demo → 拖入 `m5-model` 文件夹（含 10 文件）；
- **「文件」App**：iPhone 上从 iCloud/其他 App 拷贝亦可。
- manifest 若标 `precision` 会显示在模型行/日志里（fp32/int8 目录均可直接用，格式与 Android 相同）。

## 4. 使用与性能观测

- 选语言（六语言）→ 输入文本 → 「生成语音（边生成边播）」：AudioTrack 对应的 iOS 侧是 **AVAudioEngine + AVAudioPlayerNode 流式播放**，WAV 同时落盘 `Documents/m5-last.wav`；
- 面板指标：首块延迟 / 生成耗时 / 音频时长 / RTF / 实时倍数 / **峰值内存（phys_footprint）** / **CPU 均（getrusage 差分）**；
- 日志：`log stream --predicate 'subsystem == "com.aigump.m5demo"'` 或 Xcode console，字段与 Android 对齐：
  `[M5_LOAD]` / `[M5_PREPARE]` / `[M5_PRIME]` / `[M5_PREFILL]` / `[M5_FIRST_FRAME]` / `[M5_SEG]`（llm/codec 分段）/ `[M5_TTFA]` / `[M5_PERF] device=<机型> precision=... rtf=... first_chunk_ms=... peak_rss_mib=... cpu_pct=...`
- 已内置优化：解码器状态快照 + 音色预热（prime 一次跨请求复用）、首块 1 帧、后续 8 帧块、段尾 100ms 修剪——与 Android 终局版本一一对应。

## 5. 已知边界（demo 范围）

1. **不含文本正规化（TN）**：Android 版的 rules.json 正则引擎未移植，数字请写汉字或英文单词；
2. **不含录音克隆音色**（VoiceEncoder/182MB encoder 未移植），使用 voices.json 内置音色；
3. 框架缺失时 App 可运行但点生成会提示“缺少 OnnxRuntime xcframework”（UnavailableRunner）；
4. 线程固定 4（ORT 默认 intra-op），未做 UI 调节。

## 6. 无框架时的验证方式

```bash
./scripts/typecheck.sh    # 全源码 iOS 目标类型检查（UnavailableRunner 分支）
```

本仓库交付时的验证状态：typecheck PASS（0 error）；`Tokenizer` 与 HuggingFace `tokenizers` 黄金输出**逐位一致**（中/英样例）；分段/条件文本/NPY mmap/WAV 头部单测通过（macOS 原生跑）。真机性能数字待 xcframework 与真机到位后按第 4 节回填 `benchmarks/`。

## 7. 文件一览

```
ios/
├── project.yml                  # xcodegen 工程描述（改完跑 xcodegen generate）
├── M5Demo.xcodeproj             # 已生成，可直接 open
├── sources/
│   ├── App.swift                # 程序化 UI + 生成流程 + 性能面板 + 资源计量
│   ├── M5Engine.swift           # 生成循环/流式解码/快照预热（Android 优化版对齐移植）
│   ├── OnnxRuntimeRunner.swift  # ORT 真实实现 + 缺框架时的 UnavailableRunner
│   ├── StreamingPlayer.swift    # AVAudioEngine 流式播放
│   ├── Tokenizer.swift          # M5TK1 byte-level BPE（与 Java 版逐行等价）
│   └── Pure.swift               # 音色/条件文本/分段/模型目录/NPY/WAV
├── scripts/
│   ├── download_framework.sh    # 拉取 onnxruntime xcframework（含镜像备选）
│   └── typecheck.sh             # 无框架类型检查
└── frameworks/                  # xcframework 落地处（gitignore）
```

# m5tts-run — M5-TTS macOS 部署与首帧优化实验工程

配套课程：[ai-gump-lessons](https://github.com/ai-gump/ai-gump-lessons) 的《M5-TTS 三端部署方案》
（Apple M3 / macOS 15 / Metal + CPU 双后端的实机验证与优化记录）。

**实测数据按设备分开记录**：[华为 P40 Pro（Kirin 990）](benchmarks/huawei-p40-pro.md) · [W82 墨水屏平板（RK3566）](benchmarks/w82-eink.md)。两套数据不要混用。

**本仓库只放"轻资产"**：patch、自建 Web 测试台、测量脚本与说明。
上游代码、模型权重、运行产物都在 `.gitignore` 里排除，按下文指引自行获取/重建。

## 目录结构

```
m5tts-run/
├── patches/
│   ├── 01-m5-tts-infer-onnx-probes-and-opts.patch   # 官方 ONNX 路线的本地改动（1 文件，diff 形式）
│   └── 02-audio-cpp-m5-tts-five-knives.patch        # audio.cpp Metal 路线的本地改动（4 文件，diff 形式）
├── modified/                # 改后的完整文件（可直接覆盖回上游 clone；路径与上游一致）
│   ├── m5-tts/infer_onnx.py
│   └── audio-cpp/
│       ├── include/engine/framework/codecs/qwen3_audio_tokenizer_runtime.h
│       └── src/
│           ├── framework/codecs/qwen3_audio_tokenizer_decoder.cpp
│           └── models/m5_tts/{ar.cpp, session.cpp}
├── android/                 # 完整安卓工程（上游 M5-TTS/android 原样，含 Gradle wrapper / 打包工具 / TN 规则）
├── web/                     # 自建：流式播放测试台
│   ├── tts-test.html        #   单页测试台（GPU/CPU 双后端、语速三档、六项指标）
│   ├── proxy.py             #   同源代理（页面 + /api 路由到 gpu:8090 / cpu:8092）
│   ├── cpu_server.py        #   CPU 后端 HTTP 服务（常驻 M5Onnx，SSE 输出）
│   ├── README.md            #   启动四件套与指标说明
│   └── ref_slow.wav         #   慢速参考音频样例（8.1s，语速 3.7 字/秒）
├── measure.sh               # ONNX 路线测量脚本（线程扫描，RTF/首包）
├── measure_ac.sh            # audio.cpp 路线测量脚本（RTF/CPU）
├── ttfa.py                  # audio.cpp streaming 首块落盘监控（已证明 file sink 非增量）
├── server-config.json       # audiocpp_server 配置（⚠️ 内含绝对路径，见下文）
└── README.md
```

## 不在本仓库的部分（按说明获取）

| 目录/文件 | 大小 | 获取方式 |
|---|---|---|
| `M5-TTS/` | — | `git clone https://github.com/ASLP-lab/M5-TTS.git`；再二选一：`git apply patches/01-*.patch` **或** 用 `modified/m5-tts/` 下同名文件直接覆盖 |
| `audio.cpp/` | — | `git clone https://github.com/0xShug0/audio.cpp.git`；再二选一：`git apply patches/02-*.patch` **或** 把 `modified/audio-cpp/` 下的文件按相同相对路径覆盖 |
| `weights/` | 1.77GB | HF `ASLP-lab/M5-TTS-50M-ONNX` 全量 12 文件；`infer_onnx.py --model weights --local-files-only` |
| `m5-voiceclone-f16.gguf` | 1.0GB | HF `audio-cpp/audio.cpp-gguf` → `M5-TTS-GGUF/m5-tts-50m-voiceclone-f16.gguf`（预转换，自带 codec/tokenizer） |
| `android/models/codec_encoder.onnx` | 182MB | HF `ASLP-lab/M5-TTS-50M-ONNX` 的 `codec_encoder.onnx`；放到 `android/models/`，或构建时 `-Pm5VoiceEncoder=/绝对路径/codec_encoder.onnx` |
| `*.m5model` | 约 1GB（自行打包） | 用 `android/package_model.py` 把 `weights/` 的 4 个 ONNX + 2 个 embedding 打成外置模型包，App 内导入 |
| `outputs/` | — | 运行时自动生成（测试音频、JSON 统计、参考缓存） |

参考音频直接用 `M5-TTS/assets/aslp_default_voice.wav`（上游自带）。

## android/：完整安卓工程（上游基线 + 本地二开）

`android/` 以 [ASLP-lab/M5-TTS](https://github.com/ASLP-lab/M5-TTS) commit `4e7cc91`（2026-11-12）的 `android/` 目录为基线，完整保留 28 个原始文件（Java 源码、Gradle wrapper、`package_model.py`、文本正规化规则与测试），并做了三处本地改动：

- **六语言选择**：UI 增加语言下拉（Chinese / English / German / French / Japanese / Korean），`M5Engine.conditioned()` 参数化，不再写死 `Chinese`；
- **流式播放**：生成线程每 4 帧解码一次，`AudioTrack` 边生成边播；段尾仍保留 100ms 修剪，同时持续写 WAV 供回放/保存；
- **性能观测**：UI 底部「性能指标」显示首块延迟、生成耗时、RTF、实时倍数、峰值内存；每请求向 logcat 打 `[M5_LOAD]` / `[M5_TTFA]` / `[M5_SEG]` / `[M5_PERF]`（`adb logcat -s M5Perf`）。

其余构建事项：

- **构建环境**：JDK 17、Android SDK 35、Gradle 8.9 / AGP 8.7.3（工程已固定）、ONNX Runtime Android 1.23.2（Maven 拉取）；`minSdk 28`，只打 `arm64-v8a`；
- **构建**：`cd android && ./gradlew --no-daemon -Pm5VoiceEncoder=/绝对路径/codec_encoder.onnx :app:assembleDebug`；产物 `android/app/build/outputs/apk/debug/app-debug.apk`；
- **模型不进 APK**：主 TTS 模型用 `package_model.py` 打成外置 `.m5model`，App 内导入；APK 只内置 182MB 的参考音频 encoder（构建时按固定 SHA256 校验）；
- **上游 README 的跨目录引用**：`android/README.md` 提到的 `../README_zh.md`、`requirements.txt` 属于 M5-TTS 仓库根，本仓库没有，需要时去上游 clone 取；
- **TN 边界**：Java 版文本规范化只覆盖中文/英文，选日/韩/德/法时自动跳过 TN。详见 `android/README.md` 的《性能观测》。

## 复现步骤（M3 / macOS）

1. **Python 环境**：`uv venv --python 3.12` + `uv pip install onnxruntime numpy soundfile scipy tokenizers huggingface_hub WeTextProcessing==1.3.0`
   - `pynini==2.1.7` 在 macOS ARM 需源码编译：先 `brew install openfst`，再
     `CXXFLAGS="-I/opt/homebrew/include -std=c++17" LDFLAGS="-L/opt/homebrew/lib" uv pip install pynini==2.1.7`；
2. **编译 audio.cpp**：`brew install libomp`，然后
   `cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON -DAUDIOCPP_MODEL_SET=custom -DAUDIOCPP_MODELS=m5_tts -DAUDIOCPP_DEPLOYMENT_BUILD=ON -DOpenMP_CXX_FLAGS="-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include" -DOpenMP_CXX_LIB_NAMES=omp -DOpenMP_omp_LIBRARY=/opt/homebrew/opt/libomp/lib/libomp.dylib`
   `cmake --build build --target audiocpp_cli audiocpp_server -j 8`；
3. **启动四件套**（详见 `web/README.md`）：
   - GPU 后端 `./audio.cpp/build/bin/audiocpp_server --config server-config.json --backend metal --no-ui`
   - CPU 后端 `python web/cpu_server.py`
   - 代理/页面 `python web/proxy.py` → http://127.0.0.1:8091/tts-test.html
4. ⚠️ `server-config.json` 里的 `path` / `voice_ref` 是原作者机器的绝对路径，使用前替换成本机路径。

## 两个 patch / modified 文件做了什么（内容一致，两种分发形式）

**01（官方 ONNX 路线，`infer_onnx.py`）**
- 分段探针：`[P_REF]` 参考编码 / `[P_EMBED]` 文本 embedding / `[P_PREFILL]` prefill / `[P_FIRSTDEC]` 首块解码；
- `--ref-cache <npz>`：参考编码结果按参考音频 sha1 落盘缓存，跨进程复用；
- `--first-frames` 默认 4 → **2**（首块 2 帧恰好满足 `pending>2400` 的 emit 阈值）；
- `--left-context` 默认 8 → **4**（首块解码的参考上下文帧数；音频与 8 帧版 corr ≈0.98）。

**02（audio.cpp Metal 路线，4 文件）**
- `session.cpp`：五刀优化——① 参考编码移入 `prepare()`；② 解码器 prime 截断到 `kLeftContextFrames`（25 帧）；③ prime 移入 prepare + 首次 begin 跳过；④ 首块立即发布（不吃 100ms 尾修剪）；⑤ 解码器状态快照（prepare 存、每请求 restore）。另含分段探针（`[M5_TTFA_MS]`/`[M5_PRIME_MS]`/`[M5_FRAME_*]` 等）与每请求汇总行 `[M5_PERF]`；
- `ar.cpp`：`next_frame` 内部探针（select/project/localprefill/heads/backbone 分项计时）；
- `qwen3_audio_tokenizer_decoder.cpp` + 头文件：`StreamStateSnapshot` 快照 API（ggml_backend_tensor_get/set 拷贝状态张量）。

## 实测数据（Apple M3 / 16GB / macOS 15.7.3）

| 指标 | GPU（Metal，audio.cpp） | CPU（ONNX，infer_onnx） |
|---|---|---|
| 长文 RTF | 0.68～0.70 | 0.59～0.62 |
| CPU 峰值 | ~75% | ~400% |
| 首帧（基线） | 1.07～2.93s | 0.47～0.93s |
| 首帧（优化后） | **~0.07s**（CLI 探针）/ 74～125ms（服务） | **~0.2s**（三项参数） |
| 峰值内存 | ~2.0GB | ~2.0GB |
| 流式播放欠载 | 0 次（缓冲 150ms） | 0 次（缓冲 300ms） |

已知局限：语速偏快且无原生语速参数（但**语速随参考音频迁移**，换慢参考是最优解）；
自然度偏平（与论文 10 人听测一致——50M 容量 + 24kHz + GRPO 只优化可懂度）。

## 性能参数看哪里（自验用）

两个后端都会在服务端每请求打一行汇总，不用只盯浏览器：

| 指标 | GPU（audio.cpp / Metal） | CPU（ONNX） | 测试页 |
|---|---|---|---|
| 首帧延迟 ttft | stderr `[M5_TTFA_MS]`；SSE `speech.audio.done.timing.ttft_ms` | stderr `[CPU_PERF] ttft_ms`；SSE 同上 | 「首帧延迟」两张卡（页面实测 / 服务端） |
| 分段耗时 | `[M5_P_REF_MS]` / `[M5_P_PREFILL_MS]` / `[M5_DEC_FIRST_MS]` / `[M5_FRAME_k_MS]` | —（Python 内部分段未打点） | — |
| 音频时长 | `[M5_PERF] audio_ms` | `[CPU_PERF] audio_ms` | 音频时长 / 块数 |
| 生成耗时 | `[M5_PERF] wall_ms` | `[CPU_PERF] gen_ms` | 事件日志末行「生成耗时」 |
| RTF | `[M5_PERF] rtf`（wall / audio，越小越快） | `[CPU_PERF] rtf` | 「生成实时倍数」≈ 1 / rtf |
| 实时倍数 | `[M5_PERF] realtime` | `[CPU_PERF] realtime` | 生成实时倍数 |
| 欠载 / 缓冲 | 仅页面 | 仅页面 | 播放欠载 / 最小缓冲余量 |

采集方式：

- **GPU**：启动服务时把 stderr 落盘，每请求多一行汇总：
  `./audio.cpp/build/bin/audiocpp_server --config server-config.json --backend metal --no-ui 2>&1 | tee gpu-server.log`
  行格式：`[M5_PERF] mode=stream|batch frames=... audio_ms=... wall_ms=... rtf=... realtime=... ttfa_ms=...`
- **CPU**：`python web/cpu_server.py 2>&1 | tee cpu-server.log`
  行格式：`[CPU_PERF] ttft_ms=... gen_ms=... audio_ms=... rtf=... realtime=... chunks=... chars=... ref_cache=hit|miss`
  `ref_cache` 可用来验证参考编码缓存是否生效（首次 miss、同参考后续 hit）。
- **页面**：六张指标卡 + 事件日志；「服务端 ttft」与页面实测可交叉核对。

**对比提醒**：页面「生成实时倍数」从点击算到流结束（含首帧与网络），`[M5_PERF]` / `[CPU_PERF]` 是服务端单请求生成窗口（从进入 synthesize/generate 到产出结束），两者会略有差异；对比课程里的 RTF 时以服务端日志为准。

## 许可与合规

- M5-TTS 上游：HF 权重标 Apache-2.0（GitHub 代码仓库无 LICENSE 文件，商用逐发布件核实）；
  audio.cpp 为 MIT，GGUF 权重注意上游 codec 归属声明；
- 声音克隆仅限本人或已获授权的声音。

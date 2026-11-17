# RKNN 转换交接文档（Linux 真机执行版）

> **任务一句话**：在一台 Ubuntu 机器上，把 M5-TTS 的 `codec_decoder_cached.onnx`（声音解码器，456MB）转换成 RK3566 NPU 能跑的 `.rknn` 模型，并下载配套的 Android 版 RKNN 运行库。转换产物拿回来后，由 Mac 侧完成真机（W82 墨水屏）实测。
>
> 预计耗时：环境 20 分钟 + 转换 10～40 分钟（视机器性能）。全程只需照抄命令。

## 0. 为什么要在 Linux 上做

`rknn-toolkit2`（瑞芯微官方转换工具）**只支持 Ubuntu 18.04+（x86_64 或 arm64）**，不支持 macOS/Windows。Mac 上用 Docker 模拟又太占磁盘，所以转换步骤交给 Linux 真机。

## 1. 环境要求

| 项 | 要求 |
|---|---|
| 系统 | Ubuntu 18.04 / 20.04 / 22.04 / 24.04（x86_64 或 arm64 均可） |
| Python | 3.8 ～ 3.11（22.04 自带 3.10 正合适） |
| 磁盘 | 至少 5GB 空闲（pip 包 ~1GB + 中间产物 ~2GB） |
| 内存 | ≥ 4GB（load 456MB 模型时峰值约 2～3GB） |
| 网络 | 能访问 pypi（或国内镜像）+ github release 下载 |

## 2. 需要从 Mac 拷过去的文件（共 3 个）

放到 Linux 机器的同一个目录（下文以 `~/m5-rknn/` 为例）：

| 文件 | 大小 | 来源（Mac 上的路径） |
|---|---|---|
| `codec_decoder_cached.onnx` | 456MB | `~/Documents/Project/m5tts-run/weights/codec_decoder_cached.onnx` |
| `convert_codec.py` | 3KB | `~/Documents/Project/m5tts-run/rknn/convert_codec.py` |
| （可选）`codec_cached.json` | 3KB | `~/Documents/Project/m5tts-run/weights/codec_cached.json`（只是状态张量形状参考，转换不用） |

注意：**用 FP32 原版 onnx**，不要拿 `outputs/int8pre/` 或 `outputs/fp16/` 里的版本（量化由 RKNN 自己做）。

## 3. 安装 rknn-toolkit2

```bash
cd ~/m5-rknn
python3 -m venv venv && source venv/bin/activate
pip install -U pip

# 官方源（网络通就用这个）
pip install rknn-toolkit2

# 若 pypi 慢，用阿里云镜像
# pip install -i https://mirrors.aliyun.com/pypi/simple/ rknn-toolkit2

# 验证
python3 -c "from rknn.api import RKNN; print('ok')"
pip show rknn-toolkit2 | grep -i version   # 记下这个版本号，回传时要用
```

## 4. 执行转换

### 4.1 先转非量化版（验证连通性）

```bash
cd ~/m5-rknn && source venv/bin/activate
python3 convert_codec.py codec_decoder_cached.onnx fp
```

- 成功标志：`load_onnx ok` → `build ok (quant=False)` → `export ok`，产物 `rknn_out/codec_fp.rknn`（约 230MB）。
- **把终端完整输出保存下来**：`python3 convert_codec.py ... 2>&1 | tee fp_convert.log`

### 4.2 再转 INT8 量化版（性能目标版）

```bash
python3 convert_codec.py codec_decoder_cached.onnx i8
```

- INT8 量化需要校准数据。脚本会自动生成随机校准样本（对 codec 这种输入是 code id 的模型，随机分布足够探性能；**音质验证在板端做**）。
- 产物 `rknn_out/codec_i8.rknn`（约 60～120MB）。

### 4.3 如果报错（大概率会遇到，按此处理）

| 报错关键词 | 原因 | 处置 |
|---|---|---|
| `int64` / `Unsupported data type` | 模型的 `codes`/`offset` 输入是 int64，NPU 不支持 | 回传报错全文给 Mac 侧，我们出"int64→int32 改写版 onnx"；**不要自己改图** |
| `dynamic shape` / `-1` 维度 | `frames`/`past_length` 是动态维 | 同上回传；备用方案是出"frames 固定 8 帧"静态版 |
| `build` 阶段 OOM | 内存不足 | 加 swap：`sudo fallocate -l 8G /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile` |
| pip 装不上 | Python 版本过老 | 用 22.04 自带 3.10，或 `sudo apt install python3.10-venv` |

**报错时的黄金法则：把完整终端输出（tee 保存的 log）原样回传，不要截断。**

## 5. 下载 Android 版 RKNN 运行库（第 4 步成功后再做）

板子固件里的 librknnrt.so 是 2021 年的 1.0.0，跑不动新模型，必须带新版运行库：

```bash
# 版本号必须与第 3 步装上的 rknn-toolkit2 大版本一致（如 2.3.0）
# 到 https://github.com/rockchip-linux/rknn-toolkit2/releases 找对应版本的
# rknpu2_runtime_XXXX-android.tar.gz（aarch64）下载并解压

# github 直连慢可用代理前缀，例如：
# wget https://ghproxy.net/https://github.com/rockchip-linux/rknn-toolkit2/releases/download/v2.3.0/rknpu2_runtime_2.3.0-android.tar.gz

tar xzf rknpu2_runtime_*-android.tar.gz
# 里面取两个东西（一般在 rknn_server/android_arm64/ 或 lib/ 目录）：
#   librknnrt.so        —— Android aarch64 运行库
#   include/rknn_api.h  —— C 头文件
find . -name "librknnrt.so" -path "*android*" -o -name "rknn_api.h" | tee runtime_files.txt
```

## 6. 回传清单（做完后打包给 Mac 侧）

```
rknn_out/codec_fp.rknn        # 非量化模型（约 230MB）
rknn_out/codec_i8.rknn        # INT8 模型（若成功）
fp_convert.log                # FP 转换完整日志
i8_convert.log                # INT8 转换完整日志（若有）
librknnrt.so                  # android aarch64 运行库
rknn_api.h                    # C 头文件
toolkit2_version.txt          # pip show rknn-toolkit2 的输出
```

打包示例：`tar czf m5-rknn-results.tar.gz rknn_out/*.rknn *.log librknnrt.so rknn_api.h toolkit2_version.txt`

**验收标准**：FP 版 export 成功（ret=0）即算 Phase 1a 达标；INT8 版成功是加分项。

## 7. 拿回来之后（Mac 侧接手，不需要 Linux 参与）

1. 用 Android NDK 编译已备好的探针 `rknn/rknn_probe.c`（push 到 W82 `/data/local/tmp`，连同 `librknnrt.so` + `.rknn`）；
2. 板端直跑计时，拿到 **NPU 上 codec 单步耗时**，与 CPU 基线（W82 INT8 codec 225ms/帧）对比；
3. 数据好（≥3×）再启动 Android JNI 双后端集成。

## 附：convert_codec.py 参数说明

```
python3 convert_codec.py <源onnx路径> <fp|i8>
# 输出固定写到 ./rknn_out/；target_platform 已固定 rk3566
```

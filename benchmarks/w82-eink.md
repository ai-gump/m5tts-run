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
| 状态 | APK 已安装；模型推送/复制与生成测试进行中 |

## 实测数据

（待本次测试完成后回填；在此之前不要引用任何数字。）

## 注意

- 同一份 APK、同一份模型、同一段文本，但 RK3566 是 4×A55 的低功耗 SoC，预期 RTF 会显著高于 P40 Pro；
- 墨水屏刷新率低，UI 指标更新会滞后，性能判断以 `adb logcat -s M5Perf` 为准；
- 4GB 内存下，1.1GB 模型 + ONNX 运行时 + AudioTrack 缓冲可能接近上限，重点观察 PSS 与是否被系统杀进程。

# W82 救援与 NPU 解锁档案（任务暂停存档，2026-11-18）

> 状态：**设备当前卡开机 logo**（boot 分区被我刷入"npu status=okay"版后内核起不来）。救援方案已全部备好，按本文档任一时间可执行。

## 1. 一条命令救回设备

资产位置（都在这台 Mac）：

| 资产 | 路径 |
|---|---|
| 原始 boot 备份（64MB） | `~/Documents/Project/m5tts-run/rknn/dtb_backup/boot.img` |
| 原始 dtbo 备份（4MB） | `~/Documents/Project/m5tts-run/rknn/dtb_backup/dtbo.img` |
| 线刷工具（已编译） | `/tmp/rkdeveloptool/build/rkDevelopTool_Mac`（源码在 `/tmp/rkdeveloptool`，CMakeLists 已改好 Homebrew 路径，可随时重编） |

步骤：

1. 设备进烧写模式：**长按电源 15s 关机 → 按住音量+/reset 孔插 USB**（无按键则：关机插 USB 后长按电源 3s，放置 2 分钟等 boot 反复失败自动落 loader）；
2. 验证识别：`/tmp/rkdeveloptool/build/rkDevelopTool_Mac ld`（应列出 rockusb 设备）；
3. 写回原始 boot 并重启：

```bash
cd /tmp/rkdeveloptool/build
./rkDevelopTool_Mac ppt                 # 打印分区表，确认 boot 分区在
./rkDevelopTool_Mac wlx boot ~/Documents/Project/m5tts-run/rknn/dtb_backup/boot.img
./rkDevelopTool_Mac rd                  # 重启设备
```

4. 若 `wlx` 报错（设备在 maskrom 而非 loader，需先 `db` 一个 RK3566 loader）——从任意 RK3566 官方固件包取 `loader.bin` 后 `./rkDevelopTool_Mac db loader.bin` 再重复步骤 3。

## 2. 事故复盘（为什么 status=okay 会挂）

- 我改的内容经字节级复核**干净无误**：64MB 镜像仅 7 处 × 8 字节（7 块 base DTB 的 `npu@fde40000` 节点 `disabled\0`→`okay\0`），boot 头/内核/ramdisk 未动，dtc 反编译复验通过；
- 但厂商禁 NPU 是**整套禁**：DTB 里另有 113 个 disabled 节点未动，其中很可能包含 npu 依赖的**时钟/电源域/IO 域**。单开 status = rknpu 驱动真的 probe → 缺资源 → **内核 panic** → 卡 logo（logo 是 uboot 画的，墨水屏永久保留）；
- 教训：**跨依赖链的 DTB 开关，改 status 前必须把 clocks/power-domains/io-domains 整条链一起核对**，否则别刷。

## 3. 若未来还想解锁 NPU（正确姿势）

1. 反编译 `dtb0`（`rknn/dtb_backup/dtbs/dtb0.dtb`），检查 npu 节点引用的 clock（`cru`/`pmucru`）、power-domain 节点的 status——把整条依赖链一起 okay；
2. 校验 `assigned-clock-rates` 对应的 clock 节点是否存在于该板 DTB（被厂商物理裁掉的没法开）；
3. 先在**能进 recovery/fastboot 的镜像**上试，或刷回失败后能快速救（本文档第 1 节）；
4. NPU 硬件 + RKNN 路线本身已被证明可行：`.rknn` 模型已转出（Linux 侧 `results_20261117`，FP 235MB / INT8 120MB，frames=8/past=8），只差 `/dev/rknpu` 出现后跑探针。

## 4. 本次 W82 NPU 路线的完整时间线

| 步骤 | 结果 |
|---|---|
| 真机探测 | `/vendor` 预置 librknnrt.so（2021 年 1.0.0）+ Rockchip NNAPI HIDL 服务 |
| Linux 转 .rknn | ✅ FP + INT8 双成功（present 6D 输出 bug 用 reshape 绕过，见 Linux 报告） |
| 板上探针 | ❌ `rknn_init -1`：内核驱动未 probe |
| 根因 | DTB `npu@fde40000 status="disabled"`（7 块 base DTB 全禁） |
| 解锁尝试 | 字节手术刷入（7×8 字节）→ **内核 panic 卡 logo**（教训见第 2 节） |
| 救援 | 备份 + 工具 + 一条命令就绪（本文档第 1 节），**未执行** |

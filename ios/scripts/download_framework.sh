#!/bin/bash
# 下载 onnxruntime iOS xcframework 到 ios/frameworks/（约 60MB）
# 途径1：Mac 能翻墙/手机热点 → 直接跑本脚本
# 途径2：网络不通 → 在 Linux 机器上用同样 URL 下载后拷回，zip 放 ios/frameworks/ 再跑本脚本
set -e
VER="${1:-1.23.2}"
URLS=(
  "https://github.com/microsoft/onnxruntime/releases/download/v${VER}/onnxruntime-ios-${VER}.zip"
  "https://ghproxy.net/https://github.com/microsoft/onnxruntime/releases/download/v${VER}/onnxruntime-ios-${VER}.zip"
)
cd "$(dirname "$0")/../frameworks"
ZIP="onnxruntime-ios-${VER}.zip"
if [ ! -f "$ZIP" ]; then
  for u in "${URLS[@]}"; do
    echo "try: $u"; curl -fL --retry 2 -o "$ZIP" "$u" && break
  done
fi
[ -f "$ZIP" ] || { echo "下载失败：请在可达网络的机器上手动下载以上任一 URL，把 zip 放到 $(pwd)/ 再重跑"; exit 1; }
unzip -oq "$ZIP"
echo "OK: $(ls -d onnxruntime.xcframework 2>/dev/null || find . -name '*.xcframework' -maxdepth 2)"

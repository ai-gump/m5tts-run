#!/bin/bash
# 无 xcframework 时的全源码类型检查（UnavailableRunner 分支）
cd "$(dirname "$0")/.."
xcrun -sdk iphonesimulator swiftc -target arm64-apple-ios14.0-simulator -typecheck \
  sources/Tokenizer.swift sources/Pure.swift sources/M5Engine.swift \
  sources/OnnxRuntimeRunner.swift sources/StreamingPlayer.swift sources/App.swift
echo "TYPECHECK $([ $? -eq 0 ] && echo PASS || echo FAIL)"

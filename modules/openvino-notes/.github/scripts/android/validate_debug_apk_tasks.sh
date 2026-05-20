#!/usr/bin/env bash
set -euo pipefail

openvino_android_abi="${OPENVINO_ANDROID_ABI:-arm64-v8a}"

./gradlew \
  -PopenvinoAndroidAbi="$openvino_android_abi" \
  app:assembleDebug \
  app:assembleDebugAndroidTest \
  -m \
  --stacktrace

#!/usr/bin/env bash
set -euo pipefail

openvino_android_abi="${OPENVINO_ANDROID_ABI:-arm64-v8a}"

./gradlew \
  -PopenvinoAndroidAbi="$openvino_android_abi" \
  ai:assembleDebug \
  app:assembleDebug \
  app:assembleDebugAndroidTest \
  data:assembleDebug \
  domain:assembleDebug \
  ai:testDebugUnitTest \
  app:testDebugUnitTest \
  data:testDebugUnitTest \
  domain:testDebugUnitTest \
  --stacktrace

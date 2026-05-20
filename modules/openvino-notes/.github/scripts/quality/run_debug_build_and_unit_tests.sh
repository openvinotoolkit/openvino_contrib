#!/usr/bin/env bash
set -euo pipefail

./gradlew \
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

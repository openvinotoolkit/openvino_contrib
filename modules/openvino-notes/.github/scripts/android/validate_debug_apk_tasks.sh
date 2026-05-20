#!/usr/bin/env bash
set -euo pipefail

./gradlew \
  app:assembleDebug \
  app:assembleDebugAndroidTest \
  -m \
  --stacktrace

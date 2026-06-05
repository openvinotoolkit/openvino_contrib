#!/usr/bin/env bash
set -euo pipefail

./gradlew \
  clean \
  ai:assembleDebug \
  app:assembleDebug \
  data:assembleDebug \
  domain:assembleDebug \
  --no-build-cache \
  --rerun-tasks \
  --stacktrace

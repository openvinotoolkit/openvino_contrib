#!/usr/bin/env bash
set -euo pipefail

./gradlew \
  ai:assembleRelease \
  app:assembleRelease \
  data:assembleRelease \
  domain:assembleRelease \
  --stacktrace

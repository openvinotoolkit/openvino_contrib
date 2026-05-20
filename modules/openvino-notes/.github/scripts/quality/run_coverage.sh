#!/usr/bin/env bash
set -euo pipefail

./gradlew \
  ai:testDebugUnitTest \
  app:testDebugUnitTest \
  data:testDebugUnitTest \
  domain:testDebugUnitTest \
  koverXmlReport \
  koverVerify \
  --stacktrace

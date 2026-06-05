#!/usr/bin/env bash
set -euo pipefail

./gradlew \
  ai:lintRelease \
  app:lintRelease \
  data:lintRelease \
  domain:lintRelease \
  --stacktrace

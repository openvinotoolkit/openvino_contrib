#!/usr/bin/env bash
set -euo pipefail

./gradlew \
  ktlintCheck \
  detekt \
  ai:lintDebug \
  app:lintDebug \
  data:lintDebug \
  domain:lintDebug \
  --stacktrace

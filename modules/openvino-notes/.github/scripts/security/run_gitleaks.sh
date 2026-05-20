#!/usr/bin/env bash
set -euo pipefail

mkdir -p build/reports/gitleaks
curl -sSL "https://github.com/gitleaks/gitleaks/releases/download/v8.30.0/gitleaks_8.30.0_linux_x64.tar.gz" \
  | tar -xz gitleaks
./gitleaks detect \
  --source . \
  --report-format sarif \
  --report-path build/reports/gitleaks/gitleaks.sarif \
  --redact

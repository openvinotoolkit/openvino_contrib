#!/usr/bin/env bash
set -euo pipefail

{
  echo "## Quality Foundation"
  echo
  echo "- ktlint, detekt, and Android lint executed"
  echo "- Detekt findings are uploaded to Code Scanning"
  echo "- Lint XML/HTML reports are attached as workflow artifacts"
} >> "${GITHUB_STEP_SUMMARY:?}"

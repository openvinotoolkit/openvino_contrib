#!/usr/bin/env bash
set -euo pipefail

{
  echo "## Coverage"
  echo
  echo "Kover XML report generated at \`build/reports/kover/report.xml\`"
  echo "Coverage artifacts are attached to this workflow run."
} >> "${GITHUB_STEP_SUMMARY:?}"

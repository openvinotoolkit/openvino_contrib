#!/usr/bin/env bash
set -euo pipefail

changed_paths="${CHANGED_PATHS:-}"
file_count="$(printf '%s\n' "$changed_paths" | sed '/^$/d' | wc -l | tr -d ' ')"

{
  echo "## Preflight"
  echo
  echo "- Event: \`${EVENT_NAME:?}\`"
  echo "- Changed files: \`${file_count}\`"
  echo "- Run release: \`${RUN_RELEASE:?}\`"
  echo "- Run CodeQL: \`${RUN_CODEQL:?}\`"
  echo "- Run Android tests: \`${RUN_ANDROID_TESTS:?}\`"
  echo
  echo "### Changed Paths"
  echo '```text'
  if [[ -n "$changed_paths" ]]; then
    printf '%s\n' "$changed_paths"
  else
    echo "(none)"
  fi
  echo '```'
} >> "${GITHUB_STEP_SUMMARY:?}"

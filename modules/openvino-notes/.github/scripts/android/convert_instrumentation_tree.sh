#!/usr/bin/env bash
set -euo pipefail

results_root_dir="${RESULTS_ROOT_DIR:-android-test-results}"
emulator_suite_name="${EMULATOR_SUITE_NAME:-androidTest}"
overall_exit_code=0

if [[ ! -d "$results_root_dir" ]]; then
  echo "Results directory does not exist: $results_root_dir"
  exit 1
fi

raw_files="$(find "$results_root_dir" -name 'instrumentation-raw.txt' | sort)"
if [[ -z "$raw_files" ]]; then
  echo "No instrumentation-raw.txt files found in $results_root_dir"
  exit 1
fi

while IFS= read -r raw_file; do
  result_dir="$(dirname "$raw_file")"
  build_name="$(basename "$result_dir")"
  python3 .github/scripts/android/convert_instrumentation_to_junit.py \
    "$raw_file" \
    "$result_dir/instrumentation-results.xml" \
    "$result_dir/instrumentation.txt" \
    "${emulator_suite_name}-${build_name}"

  if rg -q 'INSTRUMENTATION_STATUS_CODE: -2|INSTRUMENTATION_FAILED:|FAILURES!!!' "$raw_file"; then
    echo "Detected instrumentation failure markers in $raw_file"
    overall_exit_code=1
  fi

  if rg -q 'tests="0"' "$result_dir/instrumentation-results.xml"; then
    echo "Detected zero executed tests in $result_dir/instrumentation-results.xml"
    overall_exit_code=1
  fi
done <<< "$raw_files"

exit "$overall_exit_code"

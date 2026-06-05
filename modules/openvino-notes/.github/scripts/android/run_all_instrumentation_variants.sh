#!/usr/bin/env bash
set -euo pipefail

apk_root_dir="${APK_ROOT_DIR:?}"
results_root_dir="${RESULTS_ROOT_DIR:-android-test-results}"
app_package="${APP_PACKAGE:-com.itlab.notes}"
test_package="${TEST_PACKAGE:-com.itlab.notes.test}"
instrumentation_runner="${INSTRUMENTATION_RUNNER:-com.itlab.notes.test/androidx.test.runner.AndroidJUnitRunner}"
export ANDROID_SERIAL="${ANDROID_SERIAL:-emulator-5554}"

adb_retry() {
  local attempt=1
  while [[ "$attempt" -le 5 ]]; do
    if adb "$@"; then
      return 0
    fi

    echo "adb command failed on attempt $attempt: adb $*"
    sleep 5
    adb start-server || true
    adb wait-for-device || true
    attempt=$((attempt + 1))
  done

  adb "$@"
}

package_installed() {
  local package_name="$1"
  adb shell pm path "$package_name" >/dev/null 2>&1
}

wait_for_boot_completed() {
  local attempt=1
  while [[ "$attempt" -le 24 ]]; do
    if adb shell getprop sys.boot_completed 2>/dev/null | tr -d '\r' | grep -q '^1$'; then
      return 0
    fi

    echo "Waiting for sys.boot_completed (attempt $attempt/24)"
    sleep 5
    adb wait-for-device || true
    attempt=$((attempt + 1))
  done

  echo "Timed out waiting for sys.boot_completed"
  adb shell getprop sys.boot_completed || true
  return 1
}

install_and_run() {
  local build_dir="$1"
  local build_name="$2"
  local result_dir="$results_root_dir/$build_name"
  local app_apk
  local test_apk
  local instrumentation_exit_code=0

  mkdir -p "$result_dir"

  app_apk="$(find "$build_dir" -name 'app-debug.apk' | head -n 1)"
  test_apk="$(find "$build_dir" -name 'app-debug-androidTest.apk' | head -n 1)"

  if [[ -z "$app_apk" || -z "$test_apk" ]]; then
    echo "Missing APK artifacts in $build_dir"
    return 1
  fi

  if package_installed "$test_package"; then
    adb shell pm clear "$test_package" >/dev/null 2>&1 || true
    adb uninstall "$test_package" >/dev/null 2>&1 || true
  fi

  if package_installed "$app_package"; then
    adb shell pm clear "$app_package" >/dev/null 2>&1 || true
    adb uninstall "$app_package" >/dev/null 2>&1 || true
  fi

  adb_retry logcat -c
  adb_retry install -r "$app_apk"
  adb_retry install -r "$test_apk"

  set +e
  adb_retry shell am instrument -w "$instrumentation_runner" | tee "$result_dir/instrumentation-raw.txt"
  instrumentation_exit_code=$?
  set -e

  adb logcat -d > "$result_dir/logcat.txt" || true

  if package_installed "$test_package"; then
    adb uninstall "$test_package" >/dev/null 2>&1 || true
  fi

  if package_installed "$app_package"; then
    adb uninstall "$app_package" >/dev/null 2>&1 || true
  fi

  return "$instrumentation_exit_code"
}

adb start-server
adb wait-for-device
wait_for_boot_completed
adb shell input keyevent 82 || true
adb_retry shell settings put global window_animation_scale 0 || true
adb_retry shell settings put global transition_animation_scale 0 || true
adb_retry shell settings put global animator_duration_scale 0 || true

mkdir -p "$results_root_dir"

mapfile -t build_dirs < <(find "$apk_root_dir" -mindepth 1 -maxdepth 1 -type d | sort)
if [[ "${#build_dirs[@]}" -eq 0 ]]; then
  echo "No build artifact directories were found in $apk_root_dir"
  exit 1
fi

overall_exit_code=0
for build_dir in "${build_dirs[@]}"; do
  build_name="$(basename "$build_dir")"
  echo "Running instrumentation for $build_name"
  if ! install_and_run "$build_dir" "$build_name"; then
    overall_exit_code=1
  fi
done

exit "$overall_exit_code"

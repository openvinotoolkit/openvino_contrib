#!/usr/bin/env bash
set -euo pipefail

apk_dir="${APK_DIR:?}"
export ANDROID_SERIAL="${ANDROID_SERIAL:-emulator-5554}"

app_apk="$(find "$apk_dir" -name 'app-debug.apk' | head -n 1)"
test_apk="$(find "$apk_dir" -name 'app-debug-androidTest.apk' | head -n 1)"

if [[ -z "$app_apk" || -z "$test_apk" ]]; then
  echo "Missing APK artifacts in $apk_dir"
  exit 1
fi

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

adb start-server
adb wait-for-device
wait_for_boot_completed
adb shell input keyevent 82 || true
adb_retry shell settings put global window_animation_scale 0 || true
adb_retry shell settings put global transition_animation_scale 0 || true
adb_retry shell settings put global animator_duration_scale 0 || true
adb_retry logcat -c
adb_retry install -r "$app_apk"
adb_retry install -r "$test_apk"

set +e
adb_retry shell am instrument -w com.itlab.notes.test/androidx.test.runner.AndroidJUnitRunner | tee instrumentation-raw.txt
instrumentation_exit_code=$?
set -e

adb logcat -d > logcat.txt || true
exit "$instrumentation_exit_code"

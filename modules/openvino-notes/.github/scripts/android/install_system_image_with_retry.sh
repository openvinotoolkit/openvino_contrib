#!/usr/bin/env bash
set -euo pipefail

system_image_package="${ANDROID_SYSTEM_IMAGE_PACKAGE:?}"
sdk_root="${ANDROID_SDK_ROOT:-${ANDROID_HOME:-}}"

if [[ -z "$sdk_root" ]]; then
  echo "ANDROID_SDK_ROOT or ANDROID_HOME must be set"
  exit 1
fi

find_sdkmanager() {
  if command -v sdkmanager >/dev/null 2>&1; then
    command -v sdkmanager
    return 0
  fi

  local candidate
  for candidate in \
    "$sdk_root/cmdline-tools/latest/bin/sdkmanager" \
    "$sdk_root/cmdline-tools/bin/sdkmanager" \
    "$sdk_root/tools/bin/sdkmanager" \
    "/usr/local/lib/android/sdk/cmdline-tools/latest/bin/sdkmanager" \
    "/usr/local/lib/android/sdk/cmdline-tools/bin/sdkmanager"
  do
    if [[ -x "$candidate" ]]; then
      echo "$candidate"
      return 0
    fi
  done

  return 1
}

sdkmanager_bin="$(find_sdkmanager || true)"
if [[ -z "$sdkmanager_bin" ]]; then
  echo "sdkmanager executable was not found"
  exit 1
fi
cleanup_partial_downloads() {
  rm -rf "$HOME/.android/cache"/* || true
  rm -rf "$sdk_root"/.android/cache/* || true
  rm -rf "$sdk_root"/temp/* || true
}

attempt=1
while [[ "$attempt" -le 3 ]]; do
  echo "Installing Android system image ($system_image_package), attempt $attempt/3"
  if "$sdkmanager_bin" --install "$system_image_package" --channel=0; then
    exit 0
  fi

  echo "System image installation failed on attempt $attempt"
  cleanup_partial_downloads
  sleep 5
  attempt=$((attempt + 1))
done

echo "Failed to install Android system image after retries: $system_image_package"
exit 1

#!/usr/bin/env bash
set -euo pipefail

set +o pipefail
yes | sdkmanager --licenses >/dev/null
set -o pipefail

packages=(
  "build-tools;${ANDROID_BUILD_TOOLS:?}"
)

platform_package="platforms;android-${ANDROID_API_LEVEL:?}"
sdk_packages="$(sdkmanager --list)"
if ! grep -Fq "${platform_package}" <<<"${sdk_packages}"; then
  fallback_platform_package="${platform_package}.0"
  if grep -Fq "${fallback_platform_package}" <<<"${sdk_packages}"; then
    platform_package="${fallback_platform_package}"
  fi
fi

packages=("${platform_package}" "${packages[@]}")

if [[ "${INSTALL_SYSTEM_IMAGE:-false}" == "true" && -n "${ANDROID_SYSTEM_IMAGE:-}" ]]; then
  packages+=("emulator")
  packages+=("${ANDROID_SYSTEM_IMAGE}")
fi

sdkmanager "${packages[@]}"

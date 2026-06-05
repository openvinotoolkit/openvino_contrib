#!/usr/bin/env bash
# Fail the release job when Android release signing secrets are not configured.
#
# Required repository secrets (Settings → Secrets and variables → Actions):
#   ANDROID_RELEASE_KEYSTORE_BASE64
#   ANDROID_RELEASE_STORE_PASSWORD
#   ANDROID_RELEASE_KEY_ALIAS
#   ANDROID_RELEASE_KEY_PASSWORD

set -euo pipefail

missing=()
[[ -z "${ANDROID_RELEASE_KEYSTORE_BASE64:-}" ]] && missing+=("ANDROID_RELEASE_KEYSTORE_BASE64")
[[ -z "${ANDROID_RELEASE_STORE_PASSWORD:-}" ]] && missing+=("ANDROID_RELEASE_STORE_PASSWORD")
[[ -z "${ANDROID_RELEASE_KEY_ALIAS:-}" ]] && missing+=("ANDROID_RELEASE_KEY_ALIAS")
[[ -z "${ANDROID_RELEASE_KEY_PASSWORD:-}" ]] && missing+=("ANDROID_RELEASE_KEY_PASSWORD")

if ((${#missing[@]} > 0)); then
  echo "Release signing is not configured. Missing repository secrets:"
  for name in "${missing[@]}"; do
    echo "  - ${name}"
  done
  echo "Add them under Settings → Secrets and variables → Actions, then re-run the workflow."
  exit 1
fi

echo "Release signing secrets are configured."

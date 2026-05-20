#!/usr/bin/env bash
# Optional CI release signing for the Android app module.
#
# Configure these repository secrets (Settings → Secrets and variables → Actions):
#   ANDROID_RELEASE_KEYSTORE_BASE64 — base64 of the .jks (e.g. base64 -w0 < release.jks)
#   ANDROID_RELEASE_STORE_PASSWORD  — keystore password
#   ANDROID_RELEASE_KEY_ALIAS         — key alias
#   ANDROID_RELEASE_KEY_PASSWORD      — key password
#
# Writes app/ci-release-signing.jks and app/keystore.properties (both gitignored).
# Called only after verify_release_signing.sh confirms all secrets are set.

set -euo pipefail

if [[ -z "${ANDROID_RELEASE_KEYSTORE_BASE64:-}" ]]; then
  echo "Release signing: ANDROID_RELEASE_KEYSTORE_BASE64 is empty."
  exit 1
fi

ROOT="${GITHUB_WORKSPACE:-.}"
APP_DIR="${ROOT}/app"
mkdir -p "${APP_DIR}"

KEY_FILE="${APP_DIR}/ci-release-signing.jks"
printf '%s' "${ANDROID_RELEASE_KEYSTORE_BASE64}" | base64 -d >"${KEY_FILE}"

PROPS_FILE="${APP_DIR}/keystore.properties"
{
  printf '%s\n' 'storeFile=ci-release-signing.jks'
  printf 'storePassword=%s\n' "${ANDROID_RELEASE_STORE_PASSWORD:-}"
  printf 'keyAlias=%s\n' "${ANDROID_RELEASE_KEY_ALIAS:-}"
  printf 'keyPassword=%s\n' "${ANDROID_RELEASE_KEY_PASSWORD:-}"
} >"${PROPS_FILE}"

echo "Release signing: wrote ${KEY_FILE} and ${PROPS_FILE}"

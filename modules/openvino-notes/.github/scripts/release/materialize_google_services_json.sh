#!/usr/bin/env bash
# Writes the Firebase Google services config needed by Android Gradle builds.
#
# Configure this repository secret (Settings -> Secrets and variables -> Actions):
#   GOOGLE_SERVICES_JSON_BASE64 - base64 of the real app/google-services.json
#
# Trusted release builds must set GOOGLE_SERVICES_JSON_REQUIRED=true.
# Pull request/quality CI may set GOOGLE_SERVICES_JSON_ALLOW_PLACEHOLDER=true to
# generate an ignored placeholder file in the workspace without storing it in git.

set -euo pipefail

ROOT="${GITHUB_WORKSPACE:-.}"
OUTPUT_PATH="${ROOT}/app/google-services.json"
REQUIRED="${GOOGLE_SERVICES_JSON_REQUIRED:-false}"
ALLOW_PLACEHOLDER="${GOOGLE_SERVICES_JSON_ALLOW_PLACEHOLDER:-false}"

if [[ -z "${GOOGLE_SERVICES_JSON_BASE64:-}" ]]; then
  if [[ "${REQUIRED}" == "true" ]]; then
    echo "Google services config is not configured. Missing repository secret:"
    echo "  - GOOGLE_SERVICES_JSON_BASE64"
    echo "Add it under Settings -> Secrets and variables -> Actions, then re-run the workflow."
    exit 1
  fi

  if [[ -f "${OUTPUT_PATH}" ]]; then
    echo "Google services config: using existing ${OUTPUT_PATH}."
    exit 0
  fi

  if [[ "${ALLOW_PLACEHOLDER}" != "true" ]]; then
    echo "Google services config is not configured."
    echo "Provide app/google-services.json locally or set GOOGLE_SERVICES_JSON_BASE64."
    exit 1
  fi

  mkdir -p "$(dirname "${OUTPUT_PATH}")"
  cat > "${OUTPUT_PATH}" <<'JSON'
{
  "project_info": {
    "project_number": "000000000000",
    "project_id": "firebase-project-id-placeholder",
    "storage_bucket": "firebase-project-id-placeholder.firebasestorage.app"
  },
  "client": [
    {
      "client_info": {
        "mobilesdk_app_id": "1:000000000000:android:0000000000000000000000",
        "android_client_info": {
          "package_name": "com.itlab.notes"
        }
      },
      "oauth_client": [
        {
          "client_id": "android-oauth-client-id-placeholder.apps.googleusercontent.com",
          "client_type": 1,
          "android_info": {
            "package_name": "com.itlab.notes",
            "certificate_hash": "0000000000000000000000000000000000000000"
          }
        },
        {
          "client_id": "web-oauth-client-id-placeholder.apps.googleusercontent.com",
          "client_type": 3
        }
      ],
      "api_key": [
        {
          "current_key": "firebase-api-key-placeholder"
        }
      ],
      "services": {
        "appinvite_service": {
          "other_platform_oauth_client": [
            {
              "client_id": "web-oauth-client-id-placeholder.apps.googleusercontent.com",
              "client_type": 3
            }
          ]
        }
      }
    }
  ],
  "configuration_version": "1"
}
JSON
  echo "Google services config: wrote CI placeholder to ${OUTPUT_PATH}."
  exit 0
fi

mkdir -p "$(dirname "${OUTPUT_PATH}")"

python3 - "${OUTPUT_PATH}" <<'PY'
import base64
import json
import os
import sys
from pathlib import Path

output_path = Path(sys.argv[1])
payload = os.environ["GOOGLE_SERVICES_JSON_BASE64"]

try:
    decoded = base64.b64decode(payload, validate=True).decode("utf-8")
except Exception as error:
    raise SystemExit(f"GOOGLE_SERVICES_JSON_BASE64 is not valid base64 UTF-8: {error}") from error

try:
    document = json.loads(decoded)
except json.JSONDecodeError as error:
    raise SystemExit(f"GOOGLE_SERVICES_JSON_BASE64 does not decode to valid JSON: {error}") from error

project_info = document.get("project_info", {})
clients = document.get("client", [])
if not isinstance(project_info, dict) or not project_info.get("project_id"):
    raise SystemExit("Decoded google-services.json is missing project_info.project_id.")
if not isinstance(clients, list) or not clients:
    raise SystemExit("Decoded google-services.json is missing at least one client entry.")

output_path.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
PY

echo "Google services config: wrote ${OUTPUT_PATH} from GOOGLE_SERVICES_JSON_BASE64."

#!/usr/bin/env bash

# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
REPOSITORY_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd -P)
PROJECT_ROOT="$REPOSITORY_ROOT/modules/openvino-notes"
WORKSPACE_ROOT=$(cd -- "$REPOSITORY_ROOT/.." && pwd -P)

if [[ -n "${OPENVINO_NOTES_STATE_ROOT:-}" ]]; then
    STATE_ROOT=$OPENVINO_NOTES_STATE_ROOT
elif [[ -f "$WORKSPACE_ROOT/RULES.md" ]]; then
    STATE_ROOT="$WORKSPACE_ROOT/builds/android/openvino-notes"
else
    STATE_ROOT="$REPOSITORY_ROOT/builds/android/openvino-notes"
fi

case "$STATE_ROOT" in
    /*) ;;
    *) STATE_ROOT="$REPOSITORY_ROOT/$STATE_ROOT" ;;
esac

if [[ -f "$WORKSPACE_ROOT/RULES.md" ]]; then
    STORAGE_BOUNDARY="$WORKSPACE_ROOT/builds/android"
else
    STORAGE_BOUNDARY=$REPOSITORY_ROOT
fi
case "/$STATE_ROOT/" in
    *"/../"*|*"/./"*)
        echo "State path must not contain dot segments: $STATE_ROOT" >&2
        exit 2
        ;;
esac
case "$STATE_ROOT" in
    "$STORAGE_BOUNDARY"|"$STORAGE_BOUNDARY"/*) ;;
    *)
        echo "State path must remain inside $STORAGE_BOUNDARY" >&2
        exit 2
        ;;
esac

WRAPPER_JAR="$PROJECT_ROOT/gradle/wrapper/gradle-wrapper.jar"
JAVA_BIN="${JAVA_HOME:+$JAVA_HOME/bin/}java"
ANDROID_SDK="${ANDROID_SDK_ROOT:-${ANDROID_HOME:-}}"

if [[ ! -f "$WRAPPER_JAR" ]]; then
    echo "Missing versioned Gradle wrapper JAR: $WRAPPER_JAR" >&2
    exit 2
fi
if ! command -v "$JAVA_BIN" >/dev/null 2>&1; then
    echo "JDK 21 is required; set JAVA_HOME or add java to PATH" >&2
    exit 3
fi
JAVA_MAJOR=$("$JAVA_BIN" -XshowSettings:properties -version 2>&1 | sed -n 's/^[[:space:]]*java.version = \([0-9][0-9]*\).*/\1/p' | head -1)
if [[ "$JAVA_MAJOR" != "21" ]]; then
    echo "JDK 21 is required; detected Java ${JAVA_MAJOR:-unknown}" >&2
    exit 4
fi
if [[ -z "$ANDROID_SDK" || ! -d "$ANDROID_SDK" ]]; then
    echo "Android SDK is required; set ANDROID_SDK_ROOT or ANDROID_HOME" >&2
    exit 5
fi
if [[ ! -d "$ANDROID_SDK/platforms/android-37.0" ]]; then
    echo "Android SDK platform 37.0 is required" >&2
    exit 6
fi

mkdir -p \
    "$STATE_ROOT/gradle-user-home" \
    "$STATE_ROOT/android-user-home" \
    "$STATE_ROOT/cache" \
    "$STATE_ROOT/tmp" \
    "$STATE_ROOT/user-home" \
    "$STATE_ROOT/project-cache" \
    "$STATE_ROOT/kotlin-project" \
    "$STATE_ROOT/gradle"

export GRADLE_USER_HOME="$STATE_ROOT/gradle-user-home"
export ANDROID_USER_HOME="$STATE_ROOT/android-user-home"
export XDG_CACHE_HOME="$STATE_ROOT/cache"
export TMPDIR="$STATE_ROOT/tmp"
export ANDROID_HOME="$ANDROID_SDK"
export ANDROID_SDK_ROOT="$ANDROID_SDK"
export JAVA_TOOL_OPTIONS="${JAVA_TOOL_OPTIONS:+$JAVA_TOOL_OPTIONS }-Duser.home=$STATE_ROOT/user-home -Djava.io.tmpdir=$STATE_ROOT/tmp"

cd "$PROJECT_ROOT"
exec "$JAVA_BIN" -classpath "$WRAPPER_JAR" org.gradle.wrapper.GradleWrapperMain \
    --project-cache-dir "$STATE_ROOT/project-cache" \
    -PopenvinoNotesBuildRoot="$STATE_ROOT/gradle" \
    -Pkotlin.project.persistent.dir="$STATE_ROOT/kotlin-project" \
    "$@"

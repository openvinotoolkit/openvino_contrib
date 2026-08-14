<!--
Copyright (C) 2018-2026 Intel Corporation
SPDX-License-Identifier: Apache-2.0
-->

# OpenVINO Notes

OpenVINO Notes is a modular Android notes application prepared for local-first
storage, Google identity and Drive synchronization, and on-device text and image
assistance with OpenVINO. The current 21-module foundation implements the domain,
Room persistence, Compose UI boundaries, account-scoped scheduling, typed adapter
contracts, and executable architecture checks. Google credentials, Drive
transport, OpenVINO runtime binaries, models, production sync, and release signing
are not bundled yet; those adapters report explicit unavailable states.

See the [documentation index](docs/README.md) for the architecture handbook and
implementation status.

## Prerequisites

- JDK 21, with `java` on `PATH`.
- Android SDK command-line tools, including `sdkmanager`.
- An Android device or emulator only if the APK will be installed.

All commands below start in the `openvino_contrib` checkout root. They keep the
SDK, Gradle downloads, caches, temporary files, reports, and APKs in the parent
workspace's `builds/android/` tree.

## Configure a Clean Build Environment

```sh
export OPENVINO_CONTRIB_ROOT="$(git rev-parse --show-toplevel)"
export OPENVINO_NOTES_WORKSPACE="$(cd "$OPENVINO_CONTRIB_ROOT/.." && pwd -P)"
export OPENVINO_NOTES_STATE_ROOT="$OPENVINO_NOTES_WORKSPACE/builds/android/openvino-notes"
export ANDROID_SDK_ROOT="$OPENVINO_NOTES_WORKSPACE/builds/android/sdk"
export ANDROID_HOME="$ANDROID_SDK_ROOT"
export GRADLE_USER_HOME="$OPENVINO_NOTES_STATE_ROOT/gradle-user-home"
export ANDROID_USER_HOME="$OPENVINO_NOTES_STATE_ROOT/android-user-home"
export XDG_CACHE_HOME="$OPENVINO_NOTES_STATE_ROOT/cache"
export TMPDIR="$OPENVINO_NOTES_STATE_ROOT/tmp"
export JAVA_TOOL_OPTIONS="-Duser.home=$OPENVINO_NOTES_STATE_ROOT/user-home -Djava.io.tmpdir=$TMPDIR"

mkdir -p "$ANDROID_SDK_ROOT" "$GRADLE_USER_HOME" "$ANDROID_USER_HOME" \
  "$XDG_CACHE_HOME" "$TMPDIR" "$OPENVINO_NOTES_STATE_ROOT/user-home" \
  "$OPENVINO_NOTES_STATE_ROOT/project-cache" \
  "$OPENVINO_NOTES_STATE_ROOT/kotlin-project" \
  "$OPENVINO_NOTES_STATE_ROOT/gradle"
```

Install the exact SDK used by the project. `sdkmanager` may itself be supplied by
an existing command-line-tools installation, but `--sdk_root` ensures downloaded
packages are written inside this workspace.

```sh
sdkmanager --sdk_root="$ANDROID_SDK_ROOT" \
  "platform-tools" "platforms;android-37.0" "build-tools;36.0.0"
yes | sdkmanager --sdk_root="$ANDROID_SDK_ROOT" --licenses
```

Verify the toolchain before building:

```sh
java -version
sdkmanager --sdk_root="$ANDROID_SDK_ROOT" --list_installed
```

## Build and Test

Run the checked-in Gradle wrapper JAR directly; no repository helper script is
required. The first invocation downloads the checksum-pinned Gradle distribution
into `GRADLE_USER_HOME`.

```sh
cd "$OPENVINO_CONTRIB_ROOT/modules/openvino-notes"

java -classpath gradle/wrapper/gradle-wrapper.jar \
  org.gradle.wrapper.GradleWrapperMain \
  --project-cache-dir "$OPENVINO_NOTES_STATE_ROOT/project-cache" \
  -PopenvinoNotesBuildRoot="$OPENVINO_NOTES_STATE_ROOT/gradle" \
  -Pkotlin.project.persistent.dir="$OPENVINO_NOTES_STATE_ROOT/kotlin-project" \
  checkArchitecture test testDebugUnitTest :app:assembleDebug
```

The default ABI is `arm64-v8a`. For an x86_64 emulator or the CI-equivalent
build, append `-PopenvinoAndroidAbi=x86_64`. The resulting APK is:

```text
builds/android/openvino-notes/gradle/app/outputs/apk/debug/app-debug.apk
```

Install it on a connected target with:

```sh
"$ANDROID_SDK_ROOT/platform-tools/adb" install -r \
  "$OPENVINO_NOTES_STATE_ROOT/gradle/app/outputs/apk/debug/app-debug.apk"
```

## Development Rules

- Use `com.openvino.notes` as the base package, Android namespace, and application identifier.
- Extend a capability through its API and owning implementation; do not add implementation-to-implementation edges.
- Treat the dependency allowlist as a ceiling: optional allowed edges need not exist.
- Keep Android and vendor types out of `:kernel`, API modules, and JVM core modules.
- Keep Room, WorkManager, Drive, OpenVINO, and DI lookups out of `:view`.
- Define all Koin bindings and ViewModel factories in `:app` only.
- Add focused tests beside implementations and reusable fakes to API `testFixtures`.
- Keep scheduled sync account-scoped; never infer a Worker's account from current UI session state.
- Persist resumable transfer progress only through `SyncTransferCheckpointPort`; never log or expose session capabilities.
- Access attachment bytes only through `AttachmentContentPort`; do not expose file paths or raw media in View APIs.
- Treat binary content as immutable by `AttachmentId`; allocate a new ID and update the Note when bytes change.
- Keep infrastructure contracts under `*.api.port`; `:view` may import consumer services but must not import ports.
- Consume attachment content in bounded chunks unless a downstream API explicitly requires a size-limited contiguous value.
- Preserve committed Room schema JSON whenever a database contract changes, and add migrations before incrementing a shipped schema.

Sync is deliberately `Blocked(NotConfigured)` until a remote-first,
revision-aware engine is implemented.

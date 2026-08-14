<!--
Copyright (C) 2018-2026 Intel Corporation
SPDX-License-Identifier: Apache-2.0
-->

# OpenVINO Notes

OpenVINO Notes is a 21-module Android architecture skeleton. It freezes feature contracts, ownership boundaries, local persistence, UI slices, and executable dependency checks so follow-up work can proceed independently. Google Identity, Drive transport, OpenVINO runtime libraries, models, credentials, and secrets are intentionally not included; their adapters return typed unavailable states.

## Build

From the repository root, use the versioned wrapper entrypoint:

```sh
./scripts/openvino_notes_gradle.sh checkArchitecture
./scripts/openvino_notes_gradle.sh test testDebugUnitTest
./scripts/openvino_notes_gradle.sh :app:assembleDebug
```

The default Android ABI is `arm64-v8a`. Use an emulator ABI explicitly when needed:

```sh
./scripts/openvino_notes_gradle.sh :app:assembleDebug -PopenvinoAndroidAbi=x86_64
```

The script requires JDK 21 and `ANDROID_SDK_ROOT` or `ANDROID_HOME`. It uses the checked-in Gradle wrapper and redirects Gradle, Android, cache, temporary, report, and APK state to `builds/android/openvino-notes/`. In a larger workspace containing `RULES.md`, that directory is created at the workspace root; in a standalone clone it is created at the repository root. Set `OPENVINO_NOTES_STATE_ROOT` to an in-workspace path to override it.

## Development Rules

- Use `com.openvino.notes` as the base package, Android namespace, and application identifier.
- Extend a capability through its API and owning implementation; do not add implementation-to-implementation edges.
- Treat the dependency allowlist as a ceiling: optional allowed edges need not exist.
- Keep Android and vendor types out of `:kernel`, API modules, and JVM core modules.
- Keep Room, WorkManager, Drive, OpenVINO, and DI lookups out of `:view`.
- Define all Koin bindings and ViewModel factories in `:app` only.
- Add focused tests beside implementations and reusable fakes to API `testFixtures`.

Sync is deliberately `Blocked(NotConfigured)` until a remote-first, revision-aware engine is implemented. See [Architecture](docs/ARCHITECTURE.md) and [Implementation Report](docs/IMPLEMENTATION_REPORT.md).

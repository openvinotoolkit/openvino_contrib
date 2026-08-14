<!--
Copyright (C) 2018-2026 Intel Corporation
SPDX-License-Identifier: Apache-2.0
-->

# OpenVINO Notes

This module is an Android architecture skeleton derived from the design documents in the workspace `dev/` directory. It establishes stable feature contracts, implementation boundaries, an Android composition root, and executable architecture checks. Google Identity, Drive transport, native OpenVINO runtime libraries, and model assets are intentionally not bundled; their adapters report typed unavailable states instead of simulating success.

## Build

Run commands from `/Volumes/SAMSUNG/repos/openvino-contrib-openvino-notes` so all generated state remains under `builds/android/openvino-notes/`:

```sh
./scripts/openvino_notes_gradle.sh checkArchitecture
./scripts/openvino_notes_gradle.sh test testDebugUnitTest
./scripts/openvino_notes_gradle.sh :app:assembleDebug
```

The APK is written to `builds/android/openvino-notes/gradle/app/outputs/apk/debug/app-debug.apk`. Generated module graphs are under `builds/android/openvino-notes/gradle/root/reports/architecture/`.

## Development Rules

- Add feature behavior to its API and implementation pair; do not introduce implementation-to-implementation dependencies.
- Keep `:kernel`, all API modules, and JVM core modules free of Android imports.
- Keep Room, WorkManager, Drive, and OpenVINO types out of `:view`.
- Bind implementations only in `:app`; Koin is limited to `:app` and `:view`.
- Add fakes to API `testFixtures` and focused regression tests beside the changed implementation.

See [Architecture](docs/ARCHITECTURE.md) and [Implementation Report](docs/IMPLEMENTATION_REPORT.md).

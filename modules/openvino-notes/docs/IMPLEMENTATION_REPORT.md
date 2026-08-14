<!--
Copyright (C) 2018-2026 Intel Corporation
SPDX-License-Identifier: Apache-2.0
-->

# Architecture Skeleton Implementation Report

## Scope

This hardening pass implements the architecture-freeze requirements for PR #1138. It retains the exact 21-module inventory while correcting dependency checks, DI ownership, consumer contracts, persistence semantics, identity UI control, sync safety, UI slicing, and build reproducibility.

The module does not include Google credentials, a Drive HTTP client, OpenVINO Android runtime binaries, native bridges, model/tokenizer bundles, production sync, release signing, or physical-device validation. Adapters report typed `NotConfigured`/unavailable results; no fake success path is presented as integration.

## Implemented Boundaries

- Gradle validates an allowed-edge subset, rejects Koin outside `:app`, and generates the actual module graph.
- Neutral `AccountKey` replaces Identity-owned account IDs in Notes, Cloud, and Sync persistence contracts.
- Notes supports complete product fields, lossless patch updates, Room round-trips, outbox transactions, revisions, conflicts, malformed changes, and tombstones.
- AI APIs expose summary, text tags, rewrite, and structured image tags. Prompts, normalization, retry, and cancellation handling stay in the text OpenVINO adapter.
- Identity launches one-shot effects through `MainActivity` and an activity-scoped Google controller with typed cancellation/result mapping.
- Unsafe upload-first sync was removed. The production service is honestly disabled; consumer state and internal checkpoints are separate.
- UI and UI state/action boundaries are organized as vertical capability slices without Koin imports.
- A relocatable repository script and dedicated JDK 21 / Android API 37 workflow run the architecture, unit, Android unit, and APK gates.

## Verification

Local verification on 2026-08-14 used JDK 21, Android API 37, Gradle 9.5.0, and the versioned repository script:

```sh
./scripts/openvino_notes_gradle.sh checkArchitecture test testDebugUnitTest :app:assembleDebug -PopenvinoAndroidAbi=x86_64 --continue
```

The command completed successfully, including APK assembly. CI verification is intentionally not claimed here: `.github/workflows/openvino_notes.yml` must run successfully on the pushed PR head.

## Follow-up Work

Separate PRs should implement approved Google Activity Result integration, token refresh and Drive Changes transport, a remote-first sync engine using the frozen revision/tombstone/reset contracts, production OpenVINO runtime/model loading, Room migrations after initial schema v1, editor/navigation UX, and target-device inference validation.

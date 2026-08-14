<!--
Copyright (C) 2018-2026 Intel Corporation
SPDX-License-Identifier: Apache-2.0
-->

# Architecture Skeleton Implementation Report

## Scope

This hardening pass implements the architecture-freeze requirements for PR #1138. It retains the exact 21-module inventory while correcting dependency checks, DI ownership, consumer contracts, persistence semantics, identity UI control, sync safety, UI slicing, and build reproducibility.

The module does not include Google credentials, a Drive HTTP client, OpenVINO Android runtime binaries, native bridges, model/tokenizer bundles, production sync, release signing, or physical-device validation. Adapters report typed `NotConfigured`/unavailable results; no fake success path is presented as integration.

## Implemented Boundaries

- Gradle validates an allowed-edge subset, rejects Koin outside `:app`, rejects infrastructure-port references in `:view`, and generates the actual module graph.
- Neutral `AccountKey` and `AccountScope` remove Identity-owned account types and the `:notes:core -> :identity:api` production edge.
- Notes supports complete product fields plus Folder lifecycle/move semantics, Room round-trips, note/folder outbox transactions, revisions, conflicts, malformed changes, and tombstones.
- Room owns account-scoped attachment files and cleanup; injected dispatchers, bounded chunk reads, and atomic chunked writes avoid attachment-sized allocations in persistence/sync. Assistant uses an explicit bounded contiguous read for image inference.
- Cloud exposes explicit initial listing/cursor bootstrap and a generic resumable upload/streamed download boundary whose terminal result includes remote metadata and revision.
- AI APIs expose summary, text tags, rewrite, and structured image tags. Prompts, normalization, retry, and cancellation handling stay in the text OpenVINO adapter.
- Identity launches one-shot effects through `MainActivity` and an activity-scoped Google controller with typed cancellation/result mapping. Startup initialization and access-token invalidation are explicit contracts.
- Unsafe upload-first sync was removed. The production service is honestly disabled; consumer state and neutral checkpoint infrastructure are separate. Work is account-scoped from scheduler name/tag through Worker input and executor invocation.
- UI and UI state/action boundaries, including folders and neutral identity loading, are organized as vertical capability slices without Koin imports.
- A relocatable repository script and dedicated JDK 21 / Android API 37 workflow run the architecture, unit, Android unit, and APK gates.

## Verification

Local verification on 2026-08-15 used JDK 21, Android API 37, Gradle 9.5.0, and the versioned repository script:

```sh
./scripts/openvino_notes_gradle.sh checkArchitecture test testDebugUnitTest :app:assembleDebug -PopenvinoAndroidAbi=x86_64
```

The command completed successfully, including APK assembly. CI verification is intentionally not claimed here: `.github/workflows/openvino_notes.yml` must run successfully on the pushed PR head.

## Follow-up Work

Separate PRs should implement approved Google Activity Result integration, token refresh and Drive transport, a remote-first sync engine using the frozen revision/tombstone/reset contracts, production OpenVINO runtime/model loading, optional Room normalization before post-v1 migrations, structured multimodal editor/navigation UX, and target-device inference validation. Text formatting remains deferred; image dimensions remain derived media metadata.

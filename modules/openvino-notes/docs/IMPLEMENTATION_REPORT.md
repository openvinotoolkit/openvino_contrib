<!--
Copyright (C) 2018-2026 Intel Corporation
SPDX-License-Identifier: Apache-2.0
-->

# Architecture Skeleton Implementation Report

## 1. Snapshot

- Target docs: workspace `dev/CODEX_PROMPT_CREATE_ARCHITECTURE_SKELETON.md` and `dev/OPENVINO_NOTES_TARGET_ARCHITECTURE.md`.
- Current-source inventory: `embedded-dev-research/openvino-notes` at `37ed7bf584c7ed7eb8abbf97fc01888e74c794d9` (`main`, 2026-05-20).
- Reference inventory: `sannarat/thesecondversionofthenewarchitectureopenvinonotes` at `06ab7568284f431b03bf1ffbadc63080b20a18a3` (`master`, 2026-08-07).

## 2. Inventory Findings

The current application used `app/domain/data/ai`, Firebase/Google Services, and Koin inside implementation code. The reference showed useful vertical UI slices, dispatcher ownership, DataStore settings, and module-graph tooling, but also coupled ViewModels directly to DAO/OpenVINO types and centralized generic database/network modules.

Accepted ideas are vertical slices, typed dispatchers/logging, DataStore, explicit graph checks, and separate feature contracts. Rejected ideas are generic `Result` carrying raw `Throwable`, `common:database`/`common:network`, implementation-aware ViewModels, service-locator DI in adapters, Firebase, and a shared Android “common” module.

## 3. Implemented Structure

The exact 21-module graph is documented in `ARCHITECTURE.md` and enforced by `checkArchitecture`. API modules contain immutable models, typed outcomes, ports, and reusable `testFixtures` fakes. JVM core modules contain notes policy, sync orchestration, assistant orchestration, and the currently unconfigured Drive adapter. Android adapters own Room v1, DataStore Preferences, Google Identity launcher boundaries, WorkManager scheduling, and independent text/image OpenVINO lifecycle shells.

`:app` is the only implementation composition root. `:view` imports feature APIs only and contains one Koin view module plus feature-oriented packages. All Gradle outputs, caches, temporary files, reports, and APKs are redirected to workspace `builds/android/openvino-notes/` by `scripts/openvino_notes_gradle.sh`.

## 4. Verification

Verified locally with JDK 21, Android API 37, AGP 9.2.0, Kotlin 2.3.21, and Gradle 9.5.0:

- `checkArchitecture` and module graph generation;
- all JVM API/core compilation and focused notes/sync/assistant tests;
- Room repository and WorkManager worker Robolectric tests on SDK 35;
- application composition smoke test;
- all Android adapters and `:view` AAR assembly;
- `:app:assembleDebug` APK assembly.

## 5. Deliberate Follow-up Work

Google sign-in credentials/scopes, Drive HTTP transport, OpenVINO Android runtime libraries, native bridges, and text/image model bundles are not present in the source snapshots and were not fabricated. The corresponding adapters therefore expose typed `NotConfigured` states. Production work must add those artifacts through repository-approved inputs, implement token refresh and Drive conflict policy, add Room migrations after schema v1, and validate native inference on the target Android ABI/device.

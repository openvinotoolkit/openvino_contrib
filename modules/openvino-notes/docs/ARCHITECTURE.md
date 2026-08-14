<!--
Copyright (C) 2018-2026 Intel Corporation
SPDX-License-Identifier: Apache-2.0
-->

# OpenVINO Notes Architecture

## Modules and Dependency Policy

The build contains exactly 21 modules: `:app`, `:view`, `:kernel`; settings, identity, notes, cloud, sync, and assistant API/implementation pairs; and independent text/image AI API/OpenVINO pairs. `:app` is the only composition root. Implementation modules use `implementation(project(...))`; `api(project(...))` is reserved for types exposed in an API module's public ABI.

`checkArchitecture` computes actual Gradle edges and requires `actualEdges - allowedEdges` to be empty. The allowlist is a ceiling, not an exact graph. It also rejects platform imports in neutral modules, implementation imports in `:view`, Koin outside `:app`, and Firebase/Google Services configuration. `generateModuleGraph` emits Mermaid and JSON from actual edges.

## Identity and Composition

`AccountKey` belongs to `:kernel` and partitions identity sessions, notes, cloud calls, and internal sync checkpoints. Notes, Cloud, and Sync APIs do not depend on Identity for this key.

Sign-in and Drive authorization are one-shot `AppUiEffect`s. `MainActivity` consumes them and invokes an activity-scoped `GoogleIdentityUiController`; the application-scoped component retains neither `Activity` nor its launcher. Typed completed, cancelled, not-configured, and failure outcomes return to `IdentityViewModel`. `AuthenticationState.Initializing` keeps startup neutral until session restoration finishes. Authentication and Drive authorization remain separate states, and `AccessTokenProvider.invalidateAccessToken()` gives Drive a cache-independent response to HTTP 401. Koin dependencies, modules, interface bindings, and ViewModel factories exist only in `:app`.

## Notes and Assistant

The immutable `Note` contract preserves ordered content items, attachment relationships, folder, tags, favorite, summary, and creation/update timestamps. `Folder` is a complete Notes capability with create, rename, reject-nonempty-delete, move-note, repository, outbox, revision, conflict, and tombstone boundaries. Core validation and patch updates preserve unspecified fields. Room initial schema v1 owns full-field mapping, local outbox transactions, revision metadata, conflict-aware remote apply, malformed records, and tombstones; remote apply never creates a local outbox loop.

`:notes:room` owns attachment metadata plus account-scoped files under `files/notes-media/`. `AttachmentContentPort` exposes no absolute paths, enforces declared sizes, performs atomic writes, and cleans obsolete content on attachment/note deletion and remote replacement. `:assistant:core` resolves attachment metadata and reads content through this port; public Assistant and View APIs never carry attachment `ByteArray` values.

Text AI exposes typed summary, text-tag, and rewrite operations; image AI returns structured tags with confidence. Prompt construction, output normalization, bounded retry, cancellation propagation, and diagnostic mapping belong to `:ai:text-openvino`. `:assistant:core` applies structured results to `summary`, deduplicated tags, or the targeted text item without rebuilding the note.

Text formatting is intentionally deferred from the frozen domain model. Image width and height are derived media metadata rather than durable `Note` state.

## Cloud Transfer Boundaries

`RemoteObjectStore.list()` performs explicit initial discovery. `RemoteChangeFeed.startCursor()` bootstraps incremental tracking, while `changes()` requires a non-null cursor; there is no hidden null-cursor mode. `ResumableTransferClient` separates session creation/resume, chunk upload, and streamed download so media never requires full-object materialization in memory. Drive currently returns typed unavailable outcomes behind these frozen contracts.

## Sync and UI

Production sync is intentionally disabled and returns `Blocked(NotConfigured)` without local or remote mutations. Consumer `SyncState` contains no cursor. Neutral `SyncCheckpointPort` lives in `:sync:api`; `:sync:android` implements it with Room, so a future JVM `:sync:core` engine never imports Android storage. WorkManager requests carry an immutable `AccountKey`, and their hashed unique names/tags support account-specific scheduling and cancellation. A worker fails closed if account identity is absent.

`:view` is split into app navigation plus identity/sign-in, notes/list, notes/editor, notes/folders, sync/status, settings, and assistant packages. Each capability owns UI state/action boundaries; screens receive ViewModels or state/callbacks and perform no DI lookup.

## Resource Ownership

`AppComposition` creates implementations. Activity controllers stay in `MainActivity`; model adapters, attachment storage, and Room components stay application-scoped. On explicit composition close, AI resources, Sync Room, and Notes Room are closed in ownership order. Google, Drive, and OpenVINO integrations remain typed `NotConfigured` until approved assets are supplied.

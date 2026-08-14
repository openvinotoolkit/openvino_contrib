<!--
Copyright (C) 2018-2026 Intel Corporation
SPDX-License-Identifier: Apache-2.0
-->

# OpenVINO Notes Architecture

## Modules and Dependency Policy

The build contains exactly 21 modules: `:app`, `:view`, `:kernel`; settings, identity, notes, cloud, sync, and assistant API/implementation pairs; and independent text/image AI API/OpenVINO pairs. `:app` is the only composition root. Implementation modules use `implementation(project(...))`; `api(project(...))` is reserved for types exposed in an API module's public ABI.

`checkArchitecture` computes actual Gradle edges and requires `actualEdges - allowedEdges` to be empty. The allowlist is a ceiling, not an exact graph: it reserves `:app -> :cloud:api` and `:app -> :cloud:drive` for production composition without activating Drive today. Sync core has no Identity edge; it uses Kernel account scope and Cloud contracts. The gate also rejects platform imports in neutral modules, implementation imports in `:view`, infrastructure `*.api.port.*` references in `:view`, Koin outside `:app`, and Firebase/Google Services configuration. `generateModuleGraph` emits Mermaid and JSON from actual edges.

## Identity and Composition

`AccountKey` belongs to `:kernel` and partitions identity sessions, notes, cloud calls, and internal sync checkpoints. Neutral `AccountScope` lets Notes observe the active account without depending on `:identity:api`; the dependency ceiling rejects that former cross-capability edge. Notes, Cloud, and Sync APIs do not depend on Identity for the partition key.

Sign-in and Drive authorization are one-shot `AppUiEffect`s. `MainActivity` consumes them and invokes an activity-scoped `GoogleIdentityUiController`; the application-scoped component retains neither `Activity` nor its launcher. Typed completed, cancelled, not-configured, and failure outcomes return to `IdentityViewModel`. `AuthenticationState.Initializing` keeps startup neutral until session restoration finishes. Authentication and Drive authorization remain separate states, and `identity.api.port.AccessTokenProvider.invalidateAccessToken()` gives Drive a cache-independent response to HTTP 401. Koin dependencies, modules, interface bindings, and ViewModel factories exist only in `:app`.

## Notes and Assistant

The immutable `Note` contract preserves ordered content items, attachment relationships, folder, tags, favorite, summary, and creation/update timestamps. `Folder` is a complete Notes capability with create, rename, reject-nonempty-delete, move-note, repository, outbox, revision, conflict, and tombstone boundaries. Legacy `Folder.metadata` is intentionally absent because no product behavior depends on it. Core validation and patch updates preserve unspecified fields. Room initial schema v1 owns full-field mapping, local outbox transactions, revision metadata, conflict-aware remote apply, malformed records, and tombstones; remote apply never creates a local outbox loop.

`:notes:room` owns attachment metadata plus account-scoped files under `files/notes-media/`. `notes.api.port.AttachmentContentPort` exposes no absolute paths. `AttachmentId` identifies immutable binary content: an identical repeat write is idempotent, different bytes are rejected, and a content change requires a new ID plus a Note update. Its stable-size `BinarySource` supports bounded offset reads; the file adapter copies 64 KiB chunks, injects `AppDispatchers`, atomically finalizes first writes, and cleans obsolete content on attachment/note deletion and remote replacement. `:assistant:core` resolves metadata and uses an explicit size-bounded `readAll` helper because image inference requires contiguous input; sync can consume chunks directly. Public Assistant and View APIs never carry attachment `ByteArray` values.

Text AI exposes typed summary, text-tag, and rewrite operations; image AI returns structured tags with confidence. Prompt construction, output normalization, bounded retry, cancellation propagation, and diagnostic mapping belong to `:ai:text-openvino`. `:assistant:core` applies structured results to `summary`, deduplicated tags, or the targeted text item without rebuilding the note.

Text formatting is intentionally deferred from the frozen domain model. Image width and height are derived media metadata rather than durable `Note` state.

## Cloud Transfer Boundaries

`RemoteObjectStore.list()` performs explicit initial discovery. `RemoteChangeFeed.startCursor()` bootstraps incremental tracking, while `changes()` requires a non-null cursor; there is no hidden null-cursor mode. `ResumableTransferClient` separates session creation/resume, chunk upload, and streamed download so media never requires full-object materialization in memory. `UploadChunkOutcome.InProgress` returns the next session offset; `Completed` returns durable `RemoteObjectMetadata` and its revision, allowing outbox acknowledgement without a lookup. Drive currently returns typed unavailable outcomes behind these frozen contracts.

## Sync and UI

Production sync is intentionally disabled and returns `Blocked(NotConfigured)` without local or remote mutations. Consumer `SyncState` contains no cursor. Neutral `SyncCheckpointPort` and `SyncTransferCheckpointPort` live in `sync.api.port`; `:sync:android` implements both with Room, so a future JVM `:sync:core` engine never imports Android storage. Transfer checkpoints retain account-scoped operation/object identity, a redacted opaque session ID, next offset, and expiry across database reopen; completion removes the transfer checkpoint before the local operation is acknowledged. WorkManager requests carry an immutable `AccountKey`, and their hashed unique names/tags support account-specific scheduling and cancellation. A worker fails closed if account identity is absent.

`:view` is split into app navigation plus identity/sign-in, notes/list, notes/editor, notes/folders, sync/status, settings, and assistant packages. Each capability owns UI state/action boundaries; screens receive ViewModels or state/callbacks and perform no DI lookup. The editor retains structured content and updates one text item without replacing images, files, links, or other text items. Consumer services remain in their API root packages, while repositories, storage, sync models, schedulers, executors, checkpoints, and credential models/providers live under `api.port` and are rejected from presentation code.

## Resource Ownership

`AppComposition` creates implementations. Activity controllers stay in `MainActivity`; model adapters, attachment storage, and Room components stay application-scoped. Version-controlled Room v1 schemas live beside the Notes and Sync adapters so later versions can add migration tests against a stable baseline. On explicit composition close, AI resources, Sync Room, and Notes Room are closed in ownership order. Google, Drive, and OpenVINO integrations remain typed `NotConfigured` until approved assets are supplied.

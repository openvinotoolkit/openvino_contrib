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

Sign-in and Drive authorization are one-shot `AppUiEffect`s. `MainActivity` consumes them and invokes an activity-scoped `GoogleIdentityUiController`; typed completed, cancelled, not-configured, and failure outcomes return to `IdentityViewModel`. Authentication and Drive authorization remain separate states. Koin dependencies, modules, interface bindings, and ViewModel factories exist only in `:app`.

## Notes and Assistant

The immutable `Note` contract preserves ordered content items, attachment relationships, folder, tags, favorite, summary, and creation/update timestamps. Core validation and patch updates preserve unspecified fields. Room initial schema v1 owns full-field mapping, local outbox transactions, revision metadata, conflict-aware remote apply, malformed records, and tombstones; remote apply never creates a local outbox loop.

Text AI exposes typed summary, text-tag, and rewrite operations; image AI returns structured tags with confidence. Prompt construction, output normalization, bounded retry, cancellation propagation, and diagnostic mapping belong to `:ai:text-openvino`. `:assistant:core` applies structured results to `summary`, deduplicated tags, or the targeted text item without rebuilding the note.

## Sync and UI

Production sync is intentionally disabled and returns `Blocked(NotConfigured)` without local or remote mutations. Consumer `SyncState` contains no cursor. `:sync:android` owns a separate Room checkpoint store for cursor, revisions, reset state, and completion time; conflict, tombstone, malformed, and remote-reset concepts are typed for the future remote-first engine.

`:view` is split into app navigation plus identity/sign-in, notes/list, notes/editor, sync/status, settings, and assistant packages. Each capability owns UI state/action boundaries; screens receive ViewModels or state/callbacks and perform no DI lookup.

## Resource Ownership

`AppComposition` creates implementations. Activity controllers stay in `MainActivity`; model adapters and Room components stay application-scoped. On explicit composition close, text AI closes first, image AI second, and Notes Room last. Google, Drive, and OpenVINO integrations remain typed `NotConfigured` until approved assets are supplied.

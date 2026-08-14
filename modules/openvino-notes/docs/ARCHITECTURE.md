<!--
Copyright (C) 2018-2026 Intel Corporation
SPDX-License-Identifier: Apache-2.0
-->

# OpenVINO Notes Architecture

## Module Graph

The project contains exactly 21 leaf modules. Dependencies point from composition/UI and implementation modules toward stable APIs; implementation modules never depend on each other.

| Layer | Modules |
|---|---|
| Composition/UI | `:app`, `:view` |
| Shared policy | `:kernel` |
| Feature APIs | `:settings:api`, `:identity:api`, `:notes:api`, `:cloud:api`, `:sync:api`, `:assistant:api`, `:ai:text-api`, `:ai:image-api` |
| JVM implementations | `:notes:core`, `:cloud:drive`, `:sync:core`, `:assistant:core` |
| Android implementations | `:settings:datastore`, `:identity:google`, `:notes:room`, `:sync:android`, `:ai:text-openvino`, `:ai:image-openvino` |

Run `checkArchitecture` to validate the exact module/edge allowlist, platform-neutral source ownership, UI isolation, Koin ownership, and absence of Firebase/Google Services. `generateModuleGraph` writes Mermaid and JSON views of the real Gradle graph.

## Ownership and Data Flow

`:notes:api` owns immutable note models and the repository/sync ports. `:notes:room` owns Room entities, database version 1, DAO mapping, and the local outbox. A local save commits the note and outbox entry in one Room transaction.

`:sync:core` consumes only identity, notes, cloud, sync, and kernel contracts. It uploads the local outbox, applies remote changes, persists a continuation token, and maps transport failures to typed sync outcomes. `:sync:android` owns unique WorkManager scheduling and its own Room sync-state database.

`:assistant:core` orchestrates the separate text and image AI APIs. Each OpenVINO adapter has its own lifecycle and currently returns `NotConfigured` until native artifacts and models are explicitly supplied.

## Composition

`:app` constructs component/factory boundaries explicitly and exposes API contracts to a single Koin application module. `:view` depends only on APIs and supplies one `viewModule` for vertical slices: identity/sign-in, notes/list, sync/status, settings, and assistant shell. UI effects contain display-safe data, never Android contexts or implementation exceptions.

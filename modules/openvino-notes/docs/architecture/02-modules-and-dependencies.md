<!--
Copyright (C) 2018-2026 Intel Corporation
SPDX-License-Identifier: Apache-2.0
-->

# Modules and Dependencies

## Module Inventory

The build contains exactly 21 Gradle modules. The table lists each module's
single responsibility and its current direct production dependencies; test-only
dependencies are omitted.

| Module | Owns | Direct dependencies |
| --- | --- | --- |
| `:kernel` | Neutral account, time, dispatch, operation ID, diagnostics | None |
| `:settings:api` | User-facing settings contract | None |
| `:settings:datastore` | DataStore settings adapter | `:settings:api` |
| `:identity:api` | Authentication, authorization, session consumer contracts | `:kernel` |
| `:identity:google` | Google identity adapter and UI controller boundary | `:identity:api` |
| `:notes:api` | Note/folder model, services, repositories, binary ports | `:kernel` |
| `:notes:core` | Notes and folder use cases plus validation | `:notes:api`, `:kernel` |
| `:notes:room` | Room entities, mapping, outbox/revisions, attachment files | `:notes:api`, `:kernel` |
| `:cloud:api` | Remote object, change-feed, and transfer contracts | `:kernel` |
| `:cloud:drive` | Drive-facing adapter placeholder | `:cloud:api`, `:identity:api`, `:kernel` |
| `:sync:api` | Consumer sync state and infrastructure checkpoint ports | `:kernel` |
| `:sync:core` | Sync orchestration boundary and disabled production service | `:sync:api`, `:kernel` |
| `:sync:android` | WorkManager scheduling and Room checkpoint storage | `:sync:api`, `:kernel` |
| `:ai:text-api` | Text summary, tags, and rewrite contracts | None |
| `:ai:text-openvino` | Text inference adapter placeholder | `:ai:text-api` |
| `:ai:image-api` | Structured image-tagging contract | None |
| `:ai:image-openvino` | Image inference adapter placeholder | `:ai:image-api` |
| `:assistant:api` | Note-level assistant consumer contract | `:notes:api` |
| `:assistant:core` | Assistant orchestration and suggestion application | Assistant, Notes, and both AI APIs |
| `:view` | Compose screens, UI state/actions, ViewModels, navigation | Consumer APIs only |
| `:app` | Android entry points, Koin bindings, implementation lifecycle | All selected APIs and implementations |

## Contract and Implementation Rules

An API module contains stable Kotlin types and interfaces. Product-facing
services live at the API package root, for example
`com.openvino.notes.notes.api.NotesService`. Repository, credential, checkpoint,
storage, and executor seams live below `*.api.port`; these are implementation
contracts and are forbidden in `:view`.

Core modules implement application policy without Android dependencies. Adapter
modules translate a platform or vendor mechanism into an API port. A core or
adapter must not depend on another adapter: for example `:sync:core` may use
`:cloud:api` and `:notes:api`, but never `:cloud:drive` or `:notes:room`.

Use Gradle `implementation(project(...))` by default. Use `api(project(...))`
only if a public ABI exposes a type owned by the target project. Do not use
transitive visibility to hide an undeclared dependency.

## Dependency Ceiling

The root build defines an `allowedEdges` set. `checkArchitecture` calculates
actual production edges and requires:

```text
actualEdges - allowedEdges = empty
```

The allowlist is a ceiling, not a required graph. It reserves approved future
edges such as `:sync:core -> :notes:api`, `:sync:core -> :cloud:api`, and
`:app -> :cloud:drive` without pretending those integrations are active. Removing
an unused actual edge is always valid; adding an edge requires architectural
justification and an explicit ceiling update.

## Automated Boundary Checks

`checkArchitecture` verifies:

- the exact 21-module inventory;
- every actual project dependency against the ceiling;
- absence of Android imports in Kernel, APIs, JVM cores, and Cloud Drive;
- absence of Room, WorkManager, OpenVINO, implementation, and `api.port` imports
  from `:view`;
- absence of Koin imports and dependencies outside `:app`;
- absence of Firebase and Google Services build configuration;
- the `com.openvino.notes` package/namespace prefix in every Kotlin source.

The same task generates `module-graph.md` and `module-graph.json` from actual
edges. Review these artifacts whenever a Gradle dependency changes.

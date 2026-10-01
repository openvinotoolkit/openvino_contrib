<!--
Copyright (C) 2018-2026 Intel Corporation
SPDX-License-Identifier: Apache-2.0
-->

# Presentation and Composition

## Presentation Slices

`:view` is divided by user capability rather than infrastructure layer: app
navigation, identity/sign-in, notes list, structured editor, folders, assistant,
sync status, and settings. Each slice owns its UI state and actions. Screens
receive a ViewModel or explicit state/callbacks; they do not perform service
location or construct infrastructure.

View may depend on consumer services and consumer models from the API roots. It
must not import:

- `*.api.port` repositories, storage, credentials, checkpoints, or executors;
- Room entities/DAOs, WorkManager, Drive adapters, or OpenVINO adapters;
- implementation modules such as `notes.room` or `cloud.drive`;
- Koin APIs or global component lookup.

This keeps UI tests independent of databases, remote providers, and native model
runtimes. Consumer-facing loading/error states should be mapped in ViewModels;
screens render data and emit actions.

## Structured Editor Rule

The editor treats `Note.content` as an ordered heterogeneous sequence. Editing
text targets a specific `ContentItemId`. It must preserve other text blocks,
images, files, links, attachment relationships, and unchanged Note fields. Adding
formatting later should extend the domain representation deliberately rather than
storing a lossy flattened copy in View state.

## Composition Root

`:app` is the only module allowed to depend on Koin. `AppComposition` constructs
the concrete object graph: Room Notes storage, Notes core, DataStore settings,
Google identity component, AI adapters, Sync persistence/service, and Assistant
core. `OpenVinoNotesApplication` publishes bindings and ViewModel factories. This
is the only place where consumer interfaces are paired with implementations.

Keeping composition explicit has two consequences:

1. Core and adapter modules remain usable in isolated JVM/Android tests without
   starting a DI container.
2. Swapping Drive, OpenVINO, or storage implementations changes App wiring, not
   ViewModels or cross-capability business policy.

No module may obtain dependencies through a global Koin lookup. New constructor
requirements must be wired in `AppComposition` and exposed through an App binding
only if a consumer actually needs them.

## Application and Activity Lifetimes

Application-scoped resources include databases, repositories, model adapters,
settings storage, sync services, and schedulers. Activity result launchers and
the Google identity UI controller are Activity-scoped and remain in
`MainActivity`. One-shot `AppUiEffect` values bridge ViewModels to those launchers
without retaining Android UI objects in application components.

The application supplies the composition-owned Worker factory to WorkManager.
Workers receive immutable account input and delegate to an executor; they do not
reach into ViewModels or the current navigation state.

## Startup and Shutdown

Startup creates the application graph, restores identity into an explicit
initializing state, initializes UI bindings, and leaves unavailable external
adapters visibly unavailable. Initialization work must not block the main thread
with model loading or remote calls.

On explicit composition close, dependents release resources before their backing
stores: AI/model resources close, then Sync Room, then Notes Room. Activity-bound
controllers close with the Activity and are not closed through the application
graph. Every newly owned resource must document its lifetime and join the
appropriate close path.

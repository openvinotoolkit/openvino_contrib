<!--
Copyright (C) 2018-2026 Intel Corporation
SPDX-License-Identifier: Apache-2.0
-->

# Testing and Evolution

## Verification Layers

Run the narrowest relevant test while developing, then the complete module gate
documented in the [module README](../../README.md). The required layers are:

1. **API and core unit tests** validate value objects, patch semantics,
   invariants, state transitions, retry/cancellation, and orchestration with
   reusable API test fixtures.
2. **Adapter tests** validate mapping, persistence, transaction ordering,
   checkpoint recovery, file behavior, and typed provider/runtime failures.
3. **Presentation tests** validate ViewModel state/actions and preservation of
   structured content without starting real infrastructure.
4. **Architecture checks** validate module inventory, dependency direction,
   platform neutrality, namespace, View isolation, and DI ownership.
5. **APK assembly and target tests** catch Android resource, ABI, packaging, and
   lifecycle failures that JVM tests cannot see.

Tests belong beside their owning module. Prefer fakes published from API
`testFixtures` when several consumers need the same contract implementation;
avoid a test utility that imports multiple concrete adapters.

## High-Risk Regression Scenarios

Changes in the relevant area must cover these behaviors explicitly:

- account A data, files, checkpoints, and work are invisible to account B;
- a partial Note patch cannot erase unspecified structured content;
- local transactions create one outbox operation and remote apply creates none;
- attachment rewrites are idempotent for identical bytes and rejected otherwise;
- resumable transfers survive database reopen and continue at the persisted
  offset without exposing the session capability;
- remote completion is persisted before an outbox item is acknowledged;
- Worker execution uses its input account, not current UI session state;
- Activity recreation does not leak or reuse an invalid identity launcher;
- inference cancellation remains cancellation and bounded reads reject oversize
  content;
- composition closes resources without double-close or use-after-close.

## Schema and Contract Evolution

API changes should add behavior without exposing adapter types. When a contract
must break, update all implementations, fakes, consumers, and architecture docs
in the same change. Do not move an infrastructure port into the consumer API to
make a dependency convenient.

For Room changes, preserve exported schema JSON, add a migration from every
supported version, and test representative full-field rows plus tombstones,
revisions, conflicts, and transfer checkpoints. A database version must not be
bumped without its migration and committed schema.

## Adding a Capability or Module

Before adding a module:

1. Confirm an existing capability cannot own the behavior cleanly.
2. Define its responsibility, public consumer surface, infrastructure ports, and
   resource lifetime.
3. Add only contract-facing edges; implementations must not depend on sibling
   implementations.
4. Update `settings.gradle.kts`, `expectedModules`, and only the required
   `allowedEdges` entries.
5. Wire the selected implementation in `:app`.
6. Add focused tests and update the appropriate handbook chapter.
7. Run `checkArchitecture`, unit tests, Android unit tests, and APK assembly.

An allowlisted edge is architectural permission, not a reason to introduce a
dependency. Prefer constructor parameters and small owned contracts over shared
service locators, cross-module implementation imports, or a growing common module.

## Review Checklist

- Does every new type have one clear owner?
- Can the changed core be tested without Android, DI, network, or native models?
- Are account, cancellation, error, and resource-lifetime semantics explicit?
- Are binary reads bounded and capabilities redacted?
- Are placeholder integrations still honest about unavailable behavior?
- Does the generated graph contain only intended actual edges?
- Do documentation and commands describe the code that is now present?

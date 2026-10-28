<!--
Copyright (C) 2018-2026 Intel Corporation
SPDX-License-Identifier: Apache-2.0
-->

# Architecture Handbook

OpenVINO Notes is organized as independently replaceable capabilities around a
small neutral kernel. Public APIs describe product behavior; implementation
modules own storage, platform, vendor, and model details; `:app` is the only
composition root. This handbook explains how those rules fit together.

## Reading Order

1. [System overview](01-system-overview.md) — goals, layers, and delivery status.
2. [Modules and dependencies](02-modules-and-dependencies.md) — all 21 modules and
   the dependency policy enforced by Gradle.
3. [Notes domain and storage](03-notes-domain-and-storage.md) — aggregates,
   invariants, persistence, attachments, and replication records.
4. [Identity and security](04-identity-and-security.md) — account scope,
   authentication, authorization, tokens, and capability handling.
5. [Cloud and synchronization](05-cloud-and-sync.md) — remote contracts,
   cursors, resumable transfers, checkpoints, and workers.
6. [AI and assistant](06-ai-and-assistant.md) — inference contracts,
   orchestration, and resource limits.
7. [Presentation and composition](07-presentation-and-composition.md) — UI slices,
   application/activity lifetimes, DI, and shutdown.
8. [Testing and evolution](08-testing-and-evolution.md) — gates, test placement,
   schema evolution, and safe extension procedure.

## Status Vocabulary

- **Implemented** means production source and tests exist in this tree.
- **Adapter placeholder** means the contract and failure mapping exist, but the
  external SDK/runtime/credential integration is deliberately absent.
- **Planned** means the frozen boundary supports later work; no successful
  runtime behavior is claimed.

The generated dependency graph is the authoritative snapshot of actual Gradle
edges. Produce it with `checkArchitecture`; its Markdown and JSON outputs are
written below the configured build root at `root/reports/architecture/`.

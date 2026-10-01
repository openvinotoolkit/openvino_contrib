<!--
Copyright (C) 2018-2026 Intel Corporation
SPDX-License-Identifier: Apache-2.0
-->

# OpenVINO Notes Documentation

This directory is the documentation entry point for the OpenVINO Notes module.
The architecture handbook describes the current source tree and marks planned
integrations explicitly; it is not a product-roadmap substitute.

## Contents

- [Architecture handbook](architecture/README.md) — design, boundaries, runtime
  flows, ownership, and extension rules.
  - [System overview](architecture/01-system-overview.md)
  - [Modules and dependencies](architecture/02-modules-and-dependencies.md)
  - [Notes domain and storage](architecture/03-notes-domain-and-storage.md)
  - [Identity and security](architecture/04-identity-and-security.md)
  - [Cloud and synchronization](architecture/05-cloud-and-sync.md)
  - [AI and assistant](architecture/06-ai-and-assistant.md)
  - [Presentation and composition](architecture/07-presentation-and-composition.md)
  - [Testing and evolution](architecture/08-testing-and-evolution.md)
- [Implementation report](IMPLEMENTATION_REPORT.md) — delivered scope,
  verification record, exclusions, and follow-up work.
- [Module README](../README.md) — project introduction and clean build commands.

Start with the system overview, then read the module-boundary rules before
changing dependencies. Use the capability-specific document for implementation
work and the testing guide before submitting a change.

<!-- Copyright (C) 2018-2026 Intel Corporation
SPDX-License-Identifier: Apache-2.0 -->

# Module CI

`overall_status.yml` is the small repository orchestrator. It uses OpenVINO
Smart CI and `.github/components.yml` to select affected modules, calls each
module workflow explicitly, and aggregates their results.

Every module owns a normal reusable workflow named `module_<name>.yml`. Build
commands, runners, containers, test commands and artifacts remain visible in
that file. Modules are independent: adding one does not add inputs or switches
to another module's workflow.

To add a module:

1. Add its category mapping to `.github/labeler.yml` and dependency entry to
   `.github/components.yml`.
2. Add `module_<name>.yml` with that module's jobs.
3. Register one conditional call in `overall_status.yml` and include it in the
   aggregate status.

The NVIDIA workflow is the first implementation. Pull requests build its unit
and functional binaries, transfer the minimal runtime through the official
artifact actions, and run both suites on the NVIDIA runner. Nightly runs add the
full functional suite and Compute Sanitizer.

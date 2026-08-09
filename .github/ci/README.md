<!-- Copyright (C) 2018-2026 Intel Corporation
SPDX-License-Identifier: Apache-2.0 -->

# Modular CI

The CI uses native reusable workflows instead of a configuration DSL:

1. `overall_status.yml` calls every module workflow as an independent peer and
   aggregates their results into `ci/gha_overall_status`.
2. `module_<name>.yml` is a short adapter that owns that module's runners,
   images, build arguments, artifacts and test policy.
3. Lifecycle workflows such as `module_openvino_plugin.yml`, together with
   `job_module_tests.yml` and `scripts/common`, implement reusable mechanisms.
   They contain no module policy.

GitHub requires `jobs.<job_id>.uses` to contain a static workflow path, so one
explicit call per module in `overall_status.yml` is unavoidable. Keeping that
registration visible is simpler than maintaining a JSON/YAML workflow DSL.

## Add a module

Add `.github/workflows/module_<name>.yml`, then register one peer job in
`overall_status.yml` and add it to the aggregate job's `needs` list. Put every
module-specific value in the module workflow; share only mechanisms that are
actually identical across modules.

## Self-hosted runners

Module tests run without network, host credentials or Docker socket inside a
non-root, read-only, capability-dropped container. Runner groups must restrict
access to this repository and the reusable test workflow.

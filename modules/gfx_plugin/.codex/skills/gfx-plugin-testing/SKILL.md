---
name: gfx-plugin-testing
description: Use when validating OpenVINO GFX plugin changes, adding regression tests, choosing the right test target, or running compare, microbench, and profiling workflows for Metal, Vulkan, Android, or Raspberry Pi paths.
---

# GFX Plugin Testing

This skill is for test selection, regression coverage, and profiling-oriented validation in `modules/gfx_plugin/`.

## Use This Skill When

- The task asks what tests to add or run.
- The task changes MLIR lowering, backend routes, properties, scheduling, caches, infer submission, or output planning.
- The user wants compare-runner, microbench, profiling-runbook, Android, or Raspberry Pi validation guidance.

## Primary References

Read in this order:

1. `docs/TESTING.md`
2. `docs/DEVELOPMENT.md`
3. `docs/USAGE.md`
4. `docs/MICROBENCH_SCHEMA.md`
5. `docs/PROFILING_RUNBOOK.md`

## Main Test Targets

- `ov_gfx_func_tests`: plugin-facing and functional behavior
- `ov_gfx_unit_tests`: focused runtime, MLIR, cache, property, and backend regressions
- `ov_gfx_runtime_micro_tests`: smaller runtime-subgraph checks
- `ov_gfx_compare_runner`: accuracy-only diff tool
- `ov_gfx_microbench`: `MB0` to `MB3` microbench and calibration workflow

## Test Selection Rules

### MLIR or transform changes

Prefer:

- `tests/unit/mlir_*_test.cpp`
- targeted IR-shape assertions
- transform-specific unit coverage

Use runtime tests only as a second layer.

### Property or plugin-surface changes

Prefer:

- `tests/unit/plugin_tests.cpp`
- property-list and device-selection checks
- compiled-model property checks when relevant

### Backend runtime changes

Prefer:

- focused unit tests first
- backend runtime tests second
- functional or end-to-end coverage when the change is externally visible

### Scheduling, partitioning, or cache changes

Inspect and extend:

- `tests/unit/gfx_parallelism_test.cpp`
- `tests/unit/gfx_stage_policy_test.cpp`
- `tests/unit/infer_submission_test.cpp`
- `tests/unit/infer_pipeline_reuse_test.cpp`
- `tests/unit/gpu_const_cache_test.cpp`
- `tests/unit/kernel_arg_reuse_test.cpp`
- `tests/unit/gpu_backend_base_test.cpp`

## Practical Command Pattern

Use the narrowest relevant binary first, then widen:

```bash
cmake --build build-gfx-plugin --target ov_gfx_unit_tests
find build-gfx-plugin -name ov_gfx_unit_tests -type f
DYLD_LIBRARY_PATH=/path/to/openvino/runtime/libs \
  <path-to-ov_gfx_unit_tests> --gtest_filter=GfxMlirTransforms.*
ctest --test-dir build-gfx-plugin --output-on-failure -L GFX
```

## Compare And Profiling Tools

- Use `ov_gfx_compare_runner` for numeric diffs, per-op narrowing, and `GFX`-only summaries.
- Do not use `ov_gfx_compare_runner` for performance numbers.
- Use `ov_gfx_microbench` plus `docs/MICROBENCH_SCHEMA.md` and `docs/PROFILING_RUNBOOK.md` for profiling triage.
- Use `tools/gfx_profile_runbook.py`, `tools/gfx_microbench_smoke.py`, `tools/gfx_calibration_diff.py`, and `tools/gfx_external_trace_summary.py` when the task is operational rather than purely code-level.

## Output Expectations

- Recommend the smallest credible test set first.
- Name the exact files or gtest suites that should move.
- When code changes alter runtime behavior, also state whether docs need to change.

---
name: gfx-plugin-testing
description: Use when validating OpenVINO GFX plugin changes, adding regression tests, choosing the right test target, or running compare, microbench, and profiling workflows for Metal, OpenCL, Android, Linux, or Raspberry Pi paths.
---

# GFX Plugin Testing

This skill is for test selection, regression coverage, and profiling-oriented
validation in `modules/gfx_plugin/`.

## Use This Skill When

- The task asks what tests to add or run.
- The task changes MLIR lowering, backend routes, properties, scheduling,
  caches, infer submission, output planning, or profiling.
- The task changes Metal placement, MPSRT metadata, MSL kernel-family routing,
  MPS/MPSGraph vendor descriptors, resource tables, storage bridges, or request
  binding.
- The task changes OpenCL source artifacts, dynamic OpenCL runtime selection,
  source-stage execution, runtime-shape allocation, chunked Concat/Split,
  boolean-buffer behavior, constant materialization, or OpenCL op coverage.
- The user wants compare-runner, microbench, profiling-runbook, Android, Linux,
  or Raspberry Pi validation guidance.

## Read First

1. `docs/TESTING.md`
2. `docs/DEVELOPMENT.md`
3. `docs/USAGE.md`
4. `docs/MICROBENCH_SCHEMA.md`
5. `docs/PROFILING_RUNBOOK.md`

## Main Targets

- `ov_gfx_func_tests`: plugin-facing and functional behavior
- `ov_gfx_unit_tests`: focused compiler, manifest, runtime, MLIR, cache,
  property, profiling, and backend regressions
- `ov_gfx_runtime_micro_tests`: smaller runtime-subgraph checks
- `ov_gfx_compare_runner`: accuracy-only diff tool
- `ov_gfx_microbench`: MB0-MB3 microbench and calibration workflow
- `ov_gfx_conv_shape_bench`: representative Conv2D compile-plus-infer probe
- standalone OpenCL Conv2D microbenches: kernel-family experiments only

## Test Selection Rules

### MLIR Or Transform Changes

Prefer:

- `tests/unit/mlir_*_test.cpp`
- `tests/unit/basic_ops_internal_test.cpp`
- transform-specific unit coverage
- IR-shape assertions when possible

Use runtime tests as a second layer.

### Property Or Plugin Surface

Prefer:

- `tests/unit/plugin_tests.cpp`
- property-list and device-selection checks
- compiled-model property checks
- dynamic-shape query/compile checks when support boundaries moved

### Compiler, Manifest, Or Runtime Descriptor

Prefer:

- `tests/unit/gpu_backend_base_test.cpp`
- `tests/unit/plugin_tests.cpp` when `query_model()` or compile behavior moved
- backend artifact tests when payload materialization reaches Metal or OpenCL
  runtime loaders
- Metal `VendorDescriptor` tests when MPS/MPSGraph payloads are routed through
  the compiler bundle or consumed by `MpsrtVendorPrimitiveStage`

### Scheduling, Partitioning, Cache, Or Infer Reuse

Inspect and extend:

- `tests/unit/gfx_parallelism_test.cpp`
- `tests/unit/gfx_stage_policy_test.cpp`
- `tests/unit/infer_submission_test.cpp`
- `tests/unit/infer_pipeline_reuse_test.cpp`
- `tests/unit/gpu_const_cache_test.cpp`
- `tests/unit/kernel_arg_reuse_test.cpp`
- `tests/unit/gpu_backend_base_test.cpp`
- `tests/unit/runtime_subgraph_test.cpp`

### Metal

For Metal placement, MPSRT, MSL source planning, or request binding:

- cover source-plan/manifest/model records in
  `tests/unit/gfx_stage_policy_test.cpp` or
  `tests/unit/basic_ops_internal_test.cpp`
- cover compiler-owned generated MSL and MPS/MPSGraph vendor-descriptor
  payloads in `tests/unit/gpu_backend_base_test.cpp`
- cover request-time execution in `tests/backends/metal/gpu_backend_test.mm`
  when the route reaches encode time
- use `tests/unit/memory_device_integration_test.mm` for memory/device
  integration behavior

### OpenCL

For OpenCL source-artifact changes:

- start with `tests/unit/gfx_opencl_source_artifacts_test.cpp`
- add `tests/unit/gpu_backend_base_test.cpp` coverage when the artifact should
  be present in the compiler executable bundle
- add runtime coverage when dynamic runtime loading, buffer binding,
  runtime-shape allocation, constant materialization, boolean storage, chunking,
  or command execution changes
- validate with an OpenCL-capable target when the change depends on the real
  runtime rather than manifest-only logic

## Command Pattern

Use the narrowest relevant binary first:

```bash
cmake --build build-gfx-plugin --target ov_gfx_unit_tests
find build-gfx-plugin -name ov_gfx_unit_tests -type f
DYLD_LIBRARY_PATH=/path/to/openvino/runtime/libs \
  <path-to-ov_gfx_unit_tests> --gtest_filter=GfxStagePolicy.*
ctest --test-dir build-gfx-plugin --output-on-failure -L GFX
```

For docs-only changes, `git diff --check` plus stale-reference grep is usually
enough unless CMake, source lists, or public properties changed.

## Compare And Profiling Tools

- Use `ov_gfx_compare_runner` for numeric diffs, per-op narrowing, real-image
  PPM checks, golden-reference comparisons, boolean tensors, and GFX-only
  summaries.
- Do not use `ov_gfx_compare_runner` for performance numbers.
- Use `ov_gfx_microbench` with `--backend auto|metal|opencl` for profiling
  triage.
- Use `ov_gfx_conv_shape_bench` for quick Conv2D compile/infer probes.
- Use standalone OpenCL Conv2D microbenches only before promoting a kernel
  family into the shared plugin contract.
- Use `tools/gfx_profile_runbook.py`, `tools/gfx_microbench_smoke.py`,
  `tools/gfx_calibration_diff.py`, and `tools/gfx_external_trace_summary.py`
  for operational profiling workflows.

## Cross-Device Rules

- macOS validates Metal.
- Android/Linux/Raspberry Pi OpenCL targets validate OpenCL source execution.
- A pass on one backend does not prove the other backend.
- Do not convert real backend failures into skips until support probing,
  lowering, runtime binding, and target execution have all been checked.

## Output Expectations

- Recommend the smallest credible test set first.
- Name exact files, targets, or gtest suites.
- State whether docs need to change when runtime behavior changes.

---
name: gfx-plugin-testing
description: Use when validating OpenVINO GFX plugin changes, adding regression tests, choosing the right test target, or running compare, microbench, and profiling workflows for Metal, Vulkan, Android, or Raspberry Pi paths.
---

# GFX Plugin Testing

This skill is for test selection, regression coverage, and profiling-oriented validation in `modules/gfx_plugin/`.

## Use This Skill When

- The task asks what tests to add or run.
- The task changes MLIR lowering, backend routes, properties, scheduling, caches, infer submission, or output planning.
- The task changes Metal placement domains, MPSRT ABI metadata, or MSL kernel-family routing.
- The task changes kernel-manifest execution kind, typed MPSRT programs, builder-plan/runtime-model records, storage bridges, vendor-stage coverage, or hybrid MPS+MSL prepared-model execution.
- The task changes Apple MPS vendor descriptors or vendor stage coverage such as Conv2D, Pool2D, Resize2D, Softmax, or TopK.
- The task changes MPSRT resource tables, external-buffer bindings, prepared resource heaps, or model/transient resource lifetimes.
- The task changes custom-kernel family classification, external-buffer ABI roles, semantic input/output roles, or dispatch-grid policy.
- The task changes Metal MSL runtime binding plans, explicit kernel-buffer order, inferred MSL buffer-argument counts, split Apple MSL/MPS source plans, compressed `MatMul` source plans, or SDPA source plans.
- The task changes SPIR-V fixed-argument adapters, compact Vulkan ABI metadata, or MLIR-side binding overrides.
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
- LLM-fusion coverage when the change introduces backend-only rewrites such as `RoPE` or `GfxSDPAWithCausalMask`

Use runtime tests only as a second layer.

### Property or plugin-surface changes

Prefer:

- `tests/unit/plugin_tests.cpp`
- property-list and device-selection checks
- compiled-model property checks when relevant
- dynamic-shape query/compile checks when support boundaries moved for `ShapeOf`, `Concat`, `Broadcast`, `Select`, `StridedSlice`, or `Range`

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

If the change is Metal-dispatch specific, also look for focused coverage around command-buffer submission, encoder reuse, and binding reuse before jumping to broader backend tests.
If the change affects Apple MPS versus Apple MSL placement, extend `tests/unit/gfx_stage_policy_test.cpp` first, then use `tests/backends/metal/gpu_backend_test.mm` for compile/prepare/encode coverage.
If the change introduces or changes a hybrid vendor-plus-custom plan, make sure coverage includes both manifest serialization in `tests/unit/gfx_stage_policy_test.cpp` and request-time execution in `tests/backends/metal/gpu_backend_test.mm`.
If the change touches Apple MPS vendor descriptors, cover descriptor acceptance/rejection and source-plan selection in `tests/unit/gfx_stage_policy_test.cpp`, then cover prepared Metal encode behavior in `tests/backends/metal/gpu_backend_test.mm` when the route is executable.
If the change touches `GfxMpsrtProgram`, generated `gfx_mpsrt_ops`, the Apple stage pipeline, builder-plan/runtime-model records, storage bridges, or runtime resource tables, also extend `tests/unit/basic_ops_internal_test.cpp` for program/model readback and `tests/unit/gfx_stage_policy_test.cpp` for serialized bridge/resource/record validation before relying on end-to-end Metal tests.
If the change touches `gfx_custom_kernel_families.*`, also cover the family id, required entry point, dispatch policy, and external-buffer role inference in `tests/unit/gfx_stage_policy_test.cpp` or `tests/unit/basic_ops_internal_test.cpp`.
If the change touches `GfxMslRuntimeBindingPlan` or MLIR-owned MSL source plans, cover role-to-argument mapping in `tests/unit/gfx_stage_policy_test.cpp`, module/call-plan materialization in `tests/unit/basic_ops_internal_test.cpp`, and request-time `kernel_buffer_order` validation in `tests/backends/metal/gpu_backend_test.mm`.
If the change touches `src/runtime/gfx_mpsrt_model.*`, cover external tensor bindings, tensor binding plans, resource lifetime classification, and ABI adaptation in `tests/unit/gfx_stage_policy_test.cpp`, then cover Metal prepared resource binding in `tests/backends/metal/gpu_backend_test.mm` when the behavior reaches encode time.
If the change touches `spirv_kernel_binding_adapter.hpp`, cover fixed-argument compact ABI attrs in `tests/unit/basic_ops_internal_test.cpp` and a Vulkan-facing plan path in `tests/unit/gfx_stage_policy_test.cpp` when route selection is affected.
If the change touches `gfx_backend_custom_kernel_adapter.*`, `gfx_stage_kernel_binding.hpp`, or `gfx_stage_runtime_values.*`, cover both the shared manifest/binding contract and at least one backend-facing Apple MSL or SPIR-V route that consumes it.

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
- Reach for `--single-op-output` when one node output needs isolated compare coverage, and `--tinyllama-prompt-inputs` when the graph expects LLM prompt-shaped integer inputs.
- Remember that the current compare runner also handles boolean outputs, prints a targeted `Select` mismatch probe, identifies outputs as `friendly_name:port`, and reports `max_index` plus reference/GFX values for the worst mismatch.
- Do not use `ov_gfx_compare_runner` for performance numbers.
- Use `ov_gfx_microbench` plus `docs/MICROBENCH_SCHEMA.md` and `docs/PROFILING_RUNBOOK.md` for profiling triage.
- Use `tests/tools/ov_gfx_conv_shape_bench.cpp` when stage-policy or Metal placement changes need a quick compile-plus-infer sample across representative Conv2D shapes.
- Use `tools/gfx_profile_runbook.py`, `tools/gfx_microbench_smoke.py`, `tools/gfx_calibration_diff.py`, and `tools/gfx_external_trace_summary.py` when the task is operational rather than purely code-level.

## Output Expectations

- Recommend the smallest credible test set first.
- Name the exact files or gtest suites that should move.
- When code changes alter runtime behavior, also state whether docs need to change.

# Testing Guide

This document summarizes how testing is organized for `modules/gfx_plugin`.

## Main Test Targets
The module now builds three main test executables from `tests/CMakeLists.txt`:
- `ov_gfx_func_tests` for plugin-facing and functional behavior
- `ov_gfx_unit_tests` for focused runtime, MLIR, cache, and backend regressions
- `ov_gfx_runtime_micro_tests` for smaller runtime-subgraph regression checks

Build it with:

```bash
cmake --build build-gfx-plugin --target ov_gfx_func_tests
cmake --build build-gfx-plugin --target ov_gfx_unit_tests ov_gfx_runtime_micro_tests ov_gfx_microbench
```

Run the CTest label:

```bash
ctest --test-dir build-gfx-plugin --output-on-failure -L GFX
```

Run the gtest binary directly:

```bash
find build-gfx-plugin -name ov_gfx_func_tests -type f
DYLD_LIBRARY_PATH=/path/to/openvino/runtime/libs \
  <path-to-ov_gfx_func_tests> --gtest_filter=MetalBasicOps.*

find build-gfx-plugin -name ov_gfx_unit_tests -type f
DYLD_LIBRARY_PATH=/path/to/openvino/runtime/libs \
  <path-to-ov_gfx_unit_tests> --gtest_filter=GfxMlirTransforms.*
```

## Test Layout
- `tests/unit/`: focused unit tests for plugin logic, MLIR lowering, and helpers
- `tests/integration/`: plugin-level integration checks
- `tests/backends/metal/`: Metal-specific runtime and behavior coverage
- `tests/backends/vulkan/`: Vulkan-specific runtime and behavior coverage
- `tests/shared_tests_instances/`: OpenVINO shared test wiring
- `tests/tools/`: helper tools such as `ov_gfx_compare_runner` and `ov_gfx_microbench`
- `bench/`: ad-hoc evaluation helpers such as `gfx_eval.py` for local or remote compare-runner orchestration
- `tools/`: developer automation scripts for profiling workflows, calibration diffs, and smoke checks

Recent additions in the tree include:
- `tests/unit/mlir_conv_parallel_test.cpp` for canonical Conv2D lowering, per-axis and combined interior-tile bounds checks, Vulkan batch-1 parallel-launch coverage, batch>1 serial-fallback coverage, im2col rewrite coverage, and absorbed-input-transform regression checks
- `tests/unit/gfx_parallelism_test.cpp` for backend-neutral parallelism-plan selection
- `tests/unit/mlir_matmul_parallel_test.cpp` for linear matmul parallel-lowering behavior
- `tests/unit/basic_ops_internal_test.cpp` for internal transform, fusion, plugin regression coverage, focused builder coverage such as ReduceSum, generated `gfx_mpsrt_ops` / `GfxMpsrtProgram` readback, Apple stage-pipeline cleanup behavior, manifest-only Apple MSL metadata checks, SPIR-V fixed-argument adapter checks, and generated MPSRT runtime-ABI call-plan readback
- `tests/unit/layout_cleanup_test.cpp` for MLIR layout-cleanup behavior, including DFL softmax expectation rewrites
- `tests/backends/vulkan/vulkan_runtime_test.cpp` for Vulkan runtime regressions
- `tests/unit/memory_device_integration_test.mm` for Metal memory/device integration behavior
- `tests/unit/infer_submission_test.cpp` for submission-window behavior
- `tests/unit/infer_pipeline_reuse_test.cpp` for reusable pipeline, prepared-input plans, prepared-output plans, and reusable host-output coverage
- `tests/unit/gfx_profiling_report_test.cpp` for compile/infer profiling JSON assembly and merge behavior
- `tests/unit/gfx_stage_policy_test.cpp` for submit-weight and route-policy heuristics
- `tests/unit/gfx_stage_policy_test.cpp` now also covers Metal placement domains, MPSRT tensor descriptors, typed program validation, stage record keys, custom-kernel family manifests, dispatch policies, semantic input/output roles, builder-plan serialization, backend-neutral runtime-model ABI adaptation, explicit external tensor bindings, Apple MPS Resize2D descriptor/source-plan coverage, resource-table external bindings, and storage-bridge descriptors for image, matrix, ndarray, and alias contracts
- `tests/backends/metal/gpu_backend_test.mm` now covers MPSRT-backed Metal compile, prepared-pipeline caching, and request-time MSL-dispatch execution
- `tests/backends/metal/gpu_backend_test.mm` now also covers manifest-driven buffer ordering, runtime-parameter roles, storage bridges, MPSRT resource tables, prepared resource heaps, vendor `MPSGemm` / Conv2D / Pool2D / Resize2D / Softmax / TopK, and hybrid multi-stage execution
- Metal MSL binding-plan coverage now includes compressed `MatMul`, SDPA and causal SDPA kernel roles, manifest-derived MSL argument counts, output-before-runtime-params ordering, scalar-param expansion, and request-time rejection of MSL dispatch stages without materialized kernel-buffer order
- SPIR-V binding-adapter coverage now keeps compact fixed-argument metadata isolated from legacy Apple MSL operand/scalar attrs
- `tests/unit/gfx_parallelism_test.cpp` now also covers Broadcom V3D-specific matmul and convolution tuning behavior
- `tests/unit/runtime_subgraph_test.cpp` for targeted runtime subgraph execution checks through `ov_gfx_runtime_micro_tests`
- `tests/unit/gpu_const_cache_test.cpp`, `tests/unit/kernel_arg_reuse_test.cpp`, and `tests/unit/gpu_backend_base_test.cpp` for cache and binding reuse layers
- `tests/tools/ov_gfx_conv_shape_bench.cpp` for representative YOLO26x Conv2D compile-plus-infer sweeps on `CPU` or `GFX`

Recent focused updates in existing tests include:
- stronger Broadcom V3D expectations for dense stride-1, huge-spatial, and ultra-dense convolution threadgroup selection in `tests/unit/gfx_parallelism_test.cpp`
- plugin property checks that `ov::available_devices` and `ov::device::id` expose numeric ids in `tests/unit/plugin_tests.cpp`
- dynamic-shape compile/query coverage for `ShapeOf` and query-time support coverage for `Concat`, `Broadcast`, `Select`, `StridedSlice`, and `Range` in `tests/unit/plugin_tests.cpp`
- MatMul-based DFL softmax expectation rewrite checks, including value-preservation against the template plugin, in `tests/unit/layout_cleanup_test.cpp`

## Typical Test Suites
Examples already present in the tree:
- `GfxBasicOps`
- `MetalBasicOps`
- `MetalPrecisionStudy`
- plugin property and backend-selection tests in `tests/unit/plugin_tests.cpp`

## When To Add Tests
Add or update tests when you change:
- supported ops or their constraints
- property parsing or supported-property lists
- backend selection behavior
- remote context / remote tensor behavior
- stage fusion behavior
- MLIR support probing
- stage policy, parallelism selection, or input-transform absorption
- Metal placement-domain selection, MPSRT ABI metadata, or MSL kernel-family mapping
- kernel-manifest execution-kind changes such as vendor-primitive versus custom-kernel routing
- generated `gfx_mpsrt_ops` / `GfxMpsrtProgram` materialization or legacy-attr cleanup rules
- generated runtime-ABI call-plan metadata, runtime resource tables, external-resource bindings, or storage-bridge contracts for Metal MPSRT execution
- Apple stage-pipeline pass sequencing or typed `gfx.mpsrt` dialect verification rules
- custom-kernel family classification, dispatch-grid policy, or external-buffer ABI role inference
- Metal MSL runtime binding plans, inferred MSL buffer-argument counts, compressed `MatMul` source plans, or SDPA source plans
- SPIR-V fixed-argument adapter metadata or Vulkan compact-ABI binding overrides
- backend-specialized routes such as chunked or direct Vulkan execution
- infer submission thresholds, submission ordering, or command-buffer lifecycle
- immutable const-cache behavior or prepared-binding reuse
- output-resolution planning, passthrough-output handling, or reusable host-output allocation
- stage-output aliasing, fused-stage lifetime reuse, or source-node-aware output routing
- stage-level profiling payloads such as `bytes_in`, `bytes_out`, `macs_est`, or `flops_est`

## Practical Strategy
- run the narrowest relevant gtest filter first
- prefer `ov_gfx_unit_tests` for focused runtime, MLIR, cache, and backend changes
- use `ov_gfx_func_tests` when validating plugin-facing behavior or full request execution
- then run the broader backend suite
- then run `ctest -L GFX` before finalizing a change

If you change backend-specific code, prefer adding at least:
- one unit or focused regression test
- one end-to-end backend test when behavior is externally visible

If you change MLIR lowering, prefer a unit test that inspects the emitted IR for the expected operation family or attributes, not only a runtime smoke test.

## Helpful Notes
- Some tests skip when the corresponding backend is unavailable on the current machine
- Metal tests require a valid Metal runtime environment
- Vulkan tests depend on Vulkan being enabled and available in the build
- `ov_gfx_compare_runner` is useful for numeric diffs and per-op narrowing when a failure is hard to isolate from the full suite; it also supports `--per-op-all`, `--reference-device`, `--reference-plugin`, and `--gfx-only`
- `ov_gfx_compare_runner` also supports boolean tensors, `--single-op-output`, `--tinyllama-prompt-inputs`, and an extra `Select` mismatch probe for harder data-dependent failures
- `ov_gfx_compare_runner` now prints outputs as `friendly_name:port` and reports `max_index`, reference value, and GFX value for the worst mismatch
- `ov_gfx_conv_shape_bench` is useful when stage-policy or Metal placement changes need a quick before/after sample on a fixed set of Conv2D shapes without running a full benchmark flow
- keep `ov_gfx_compare_runner` accuracy-only and use `benchmark_app` for perf
- use `ov_gfx_microbench` for `MB0` to `MB3`, calibration artifacts, and profiling triage rather than for acceptance perf numbers
- for the full profiling workflow and external tracing commands, use `PROFILING_RUNBOOK.md`
- for the microbench JSON and calibration-artifact contract, use `MICROBENCH_SCHEMA.md`
- for optional automation around those flows, use `tools/gfx_profile_runbook.py`, `tools/gfx_microbench_smoke.py`, `tools/gfx_calibration_diff.py`, and `tools/gfx_external_trace_summary.py`
- Reuse-related regressions are often easier to catch with focused unit tests than with full end-to-end backend suites

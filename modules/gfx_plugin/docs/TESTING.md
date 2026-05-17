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
- `tests/unit/mlir_conv_parallel_test.cpp` for canonical Conv2D lowering, per-axis and combined interior-tile bounds checks, Vulkan batch-1 parallel-launch coverage, batch>1 serial-fallback coverage, im2col rewrite coverage, MaxPool2D/AvgPool2D parallel-dispatch regression checks, and absorbed-input-transform regression checks
- `tests/unit/gfx_parallelism_test.cpp` for backend-neutral parallelism-plan selection
- `tests/unit/mlir_matmul_parallel_test.cpp` for linear matmul parallel-lowering behavior
- `tests/unit/basic_ops_internal_test.cpp` for internal transform, fusion, plugin regression coverage, focused builder coverage such as ReduceSum, generated `gfx_mpsrt_ops` / `GfxMpsrtProgram` readback, Apple stage-pipeline cleanup behavior, manifest-only Apple MSL metadata checks, SPIR-V fixed-argument adapter checks, and typed MPSRT builder-plan/runtime-model readback
- `tests/unit/layout_cleanup_test.cpp` for MLIR layout-cleanup behavior, including DFL softmax expectation rewrites
- `tests/backends/vulkan/vulkan_runtime_test.cpp` for Vulkan runtime regressions
- `tests/unit/memory_device_integration_test.mm` for Metal memory/device integration behavior
- `tests/unit/infer_submission_test.cpp` for submission-window behavior, including common workload-profile tuning and device-family MAC budget scaling
- `tests/unit/infer_pipeline_reuse_test.cpp` for reusable pipeline, prepared-input plans, prepared-output plans, and reusable host-output coverage
- `tests/unit/gfx_profiling_report_test.cpp` for compile/infer profiling JSON assembly and merge behavior
- `tests/unit/gfx_stage_policy_test.cpp` for submit-weight and route-policy heuristics
- `tests/unit/gfx_stage_policy_test.cpp` now also covers Metal placement domains, MPSRT tensor descriptors, typed program validation, stage record keys, custom-kernel family manifests, dispatch policies, semantic input/output roles, builder-plan serialization, backend-neutral runtime-model ABI adaptation, explicit external tensor bindings, Apple MPS Resize2D descriptor/source-plan coverage, resource-table external bindings, and storage-bridge descriptors for image, matrix, ndarray, and alias contracts
- `tests/unit/gfx_stage_policy_test.cpp` also locks the accepted MPS-first image policy: supported Metal Conv/GroupConv/Pool placement stays on MPSRT/MPSImage, while the diagnostic-only `GFX_DIAGNOSTIC_F32_MPS_IMAGE` path is only an extra localization route through the same stage policy
- `tests/unit/gfx_stage_policy_test.cpp` also locks single-stage Apple MSL dispatch lifetime: `ConstTensor` and runtime-parameter roles stay external runtime-ABI resources, while backend tests cover model-owned const resources for vendor/typed MPSRT plans
- `tests/backends/metal/gpu_backend_test.mm` now covers MPSRT-backed Metal compile, prepared-pipeline caching, and request-time MSL-dispatch execution
- `tests/backends/metal/gpu_backend_test.mm` now also covers manifest-driven buffer ordering, runtime-parameter roles, storage bridges, MPSRT resource tables, prepared resource heaps, vendor `MPSGemm` / Conv2D / Pool2D / Resize2D / Softmax / TopK / SDPA, MPSGraph executable paths, and hybrid multi-stage execution
- Metal MSL binding-plan coverage now includes compressed `MatMul`, SDPA and causal SDPA kernel roles, manifest-derived MSL argument counts, output-before-runtime-params ordering, scalar-param expansion, and request-time rejection of MSL dispatch stages without materialized kernel-buffer order
- SPIR-V binding-adapter coverage now keeps compact ABI reconstruction manifest-driven, while no-manifest legacy operand/scalar attrs and `gfx.fixed_arg_count` are rejected instead of being used as runtime metadata
- `tests/unit/gfx_parallelism_test.cpp` now also covers Broadcom V3D-specific matmul and convolution tuning behavior, including huge-spatial pointwise/light Conv2D selection of occupancy-aware 64-thread groups and ultra-dense Conv capping at 128 threads per group
- `tests/unit/mlir_conv_parallel_test.cpp` now locks Pool2D away from the legacy serial MLIR route: MaxPool2D and AvgPool2D builders must produce `scf.parallel`, `gfx.parallel_loop_dims=5`, and a 64-thread dispatch shape before backend codegen.
- `tests/gfx_accuracy_tolerance.hpp` keeps shared and compare-runner tolerances precision-aware so FP16 GFX runs are not evaluated with FP32-only epsilon floors unless a test explicitly overrides them
- `tests/unit/runtime_subgraph_test.cpp` for targeted runtime subgraph execution checks through `ov_gfx_runtime_micro_tests`
- `tests/unit/gpu_const_cache_test.cpp`, `tests/unit/kernel_arg_reuse_test.cpp`, and `tests/unit/gpu_backend_base_test.cpp` for cache and binding reuse layers
- `tests/tools/ov_gfx_conv_shape_bench.cpp` builds as `ov_gfx_conv_shape_bench` and provides representative YOLO26x Conv2D compile-plus-infer sweeps. Use `--device GFX` for plugin inference checks; `CPU` is only a separate performance orienter, not a GFX fallback. Use `--case SUBSTRING` or `--list-cases` for bounded Android/RPi smoke runs before the full benchmark route.

Recent focused updates in existing tests include:
- stronger Broadcom V3D expectations for dense stride-1, huge-spatial, and ultra-dense convolution threadgroup selection in `tests/unit/gfx_parallelism_test.cpp`
- Pool2D parallel dispatch expectations in `tests/unit/mlir_conv_parallel_test.cpp`, including the guard against reverting MaxPool2D/AvgPool2D to single-dispatch serial lowering
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
- typed MPSRT builder-plan/runtime-model metadata, runtime resource tables, external-resource bindings, or storage-bridge contracts for Metal MPSRT execution
- Apple stage-pipeline pass sequencing or typed `gfx.mpsrt` dialect verification rules
- custom-kernel family classification, dispatch-grid policy, or external-buffer ABI role inference
- Metal MSL runtime binding plans, inferred MSL buffer-argument counts, compressed `MatMul` source plans, or SDPA source plans
- SPIR-V fixed-argument adapter metadata, Vulkan compact-ABI binding overrides, or module-cache reuse of lowered launch metadata
- backend-specialized routes such as chunked or direct Vulkan execution; Conv2D and GroupConv regressions should exercise the shared MLIR/SPIR-V path, not a separate Vulkan direct/chunked path
- infer submission thresholds, submission ordering, command-buffer lifecycle, or device-family budget inheritance from the common workload profile
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

For cross-device acceptance, use one logical test contract but treat backend results separately:
- Mac validates the Metal/MPS/MSL route.
- Android and Raspberry Pi validate the Vulkan/SPIR-V route on real devices.
- A Mac pass does not prove Android/RPi correctness, and an Android/RPi failure must not be converted into a skip just because the Metal route passes.
- If Android and Raspberry Pi fail with the same compare-runner mismatch, fix the shared MLIR/SPIR-V path first before adding backend-specific runtime code.
- Do not use runtime environment switches to keep alternate routes alive. A new route is acceptable only when it replaces the old path or is selected by the normal compile-time/backend planning contract.
- For Mac `f32` MPSImage experiments, use `ov_gfx_compare_runner --diagnostic-f32-mps-image` so the route goes through the normal compile property, stage planner, and MPSRT path. Do not turn that diagnostic flag into production placement until the required TEMPLATE/reference model-level gate passes.
- Keep the Conv2D `ExplicitPadding` -> `AutoPadValid` sequence as a required regression probe for Vulkan serial fallback ABI; it must stay a real pass rather than a skip on supported real devices.

If you change backend-specific code, prefer adding at least:
- one unit or focused regression test
- one end-to-end backend test when behavior is externally visible

If you change MLIR lowering, prefer a unit test that inspects the emitted IR for the expected operation family or attributes, not only a runtime smoke test.

## Helpful Notes
- Some tests skip when the corresponding backend is unavailable on the current machine
- Metal tests require a valid Metal runtime environment
- Vulkan tests depend on Vulkan being enabled and available in the build
- `ov_gfx_compare_runner` is useful for numeric diffs and per-op narrowing when a failure is hard to isolate from the full suite; it also supports `--per-op-all`, `--reference-device`, `--reference-plugin`, `--input-image`, `--dump-reference-dir`, `--golden-dir`, `--gfx-inference-precision f16|f32`, `--per-op-input-mode`, `--per-op-generated-inputs`, `--per-op-recursive-limit`, `--per-op-recursive-trace`, and `--gfx-only`
- Accuracy thresholds are precision-aware by default, following the same principle as OpenVINO shared/Intel GPU tests: the reference device remains `TEMPLATE`, but the tolerance floor is derived from the model/output types and the actual GFX inference precision. FP16 GFX runs must not be judged with FP32-only epsilons unless the command explicitly passes strict overrides such as `--abs-threshold 1e-6 --rel-threshold 1e-6`. Manual thresholds are still hard gates and are not relaxed by the default policy.
- YOLO26x accuracy/data-dependence gates should use at least three reproducible real RGB images through repeated `--input-image` arguments rather than relying only on `--random-seed-count`. The runner accepts portable Netpbm PPM (`P6`/`P3`) so the same files can be staged by `bench/gfx_eval.py` on macOS, Android, and SSH/RPi without adding OpenCV/ImageIO dependencies to the test binary. The preprocessing contract is RGB resize to the model's rank-4 batch-1 NCHW or NHWC input, normalized to `[0, 1]` for floating-point tensors and `[0, 255]` for `u8`.
- Slow or low-power targets may compare against golden reference tensors produced by the same runner instead of running the reference backend on the device. Generate them with `--dump-reference-dir DIR --reference-device TEMPLATE --reference-plugin ...` on a machine that can finish the TEMPLATE inference, stage the resulting small `output_*.ovtensor` files with the same PPM inputs, and run the target with `--golden-dir DIR`. This is still a TEMPLATE/reference accuracy gate and the target run compiles/executes only `GFX`; it must not become a plugin CPU fallback, a test skip, or a backend-specific expected-output shortcut. For repeated `--input-image` runs, the runner writes and reads per-image subdirectories named `image_0`, `image_1`, and so on.
- `--gfx-inference-precision f16|f32` is a compile-property diagnostic for the GFX model only. It must not be used as a runtime switch, test skip, or permission to demote MPS-covered stages to MSL. If `f32` does not improve a real-image mismatch, keep the next fix in the MPSRT storage/layout/descriptor/numeric contract or another typed MPS-family route rather than adding a backend-specific skip.
- Full-model compare output prints `MAX_ABS` in addition to `FIRST_MISMATCH`. Use `FIRST_MISMATCH` to find the first threshold violation and `MAX_ABS` to find the element that dominates model-level quality, especially for YOLO decode paths where bbox arithmetic can amplify a smaller upstream score/box drift.
- Full-graph `--per-op-all` scans are thresholded gates: a clean scan ends with `PER_OP_MATCH max_abs=... max_rel=... tolerance_violations=0`, while drift emits `PER_OP_FIRST_MISMATCH` plus `PER_OP_MISMATCH` and exits non-zero. Failure is decided per element with the usual `abs <= threshold OR rel <= threshold` acceptance rule; independent global `max_abs` and `max_rel` counters from different elements are reported for profiling, not used as a combined failure predicate. `bench/gfx_eval.py` preserves this as `mode=per_op_mismatch` on host, Android, and SSH transports.
- `--per-op-all` compares observable graph ports only: model outputs and ports consumed by downstream ops. Unused auxiliary outputs such as unconsumed `MaxPool` indices are not part of the production accuracy contract and must not be reported as per-op mismatches.
- Per-op, `--per-op-all`, and `--single-op-output` GFX compilations disable fusion with `GFX_ENABLE_FUSION=false` through the normal compile-property path. This keeps artificial diagnostic models aligned with original op boundaries and leaves production/full-graph fusion unchanged.
- Vulkan Conv2D retry paths must use the same manifest-backed adapter contract as ordinary custom kernels. If a retry narrows the final SPIR-V signature, express that as an explicit direct IO ABI and rebuild the final `gfx.kernel_operand_*` adapter attrs; do not keep wider ConvParams extras or infer the runtime binding from observed shader binding count.
- For slow remote targets, use `--per-op-input-mode generated` only after the failing op has already been identified. This avoids spending the whole timeout in a reference upstream subgraph while still comparing the isolated op against the same reference backend on deterministic tensors with the original shapes and types. `--per-op-generated-inputs` is the compatibility alias.
- `--per-op-input-mode gfx-recursive` is the single GFX upstream-materialization mode for per-op checks. It replaces the removed monolithic `gfx-upstream` path by materializing producers one stage at a time through a shared cache. Host materialization here is compare-runner-only test input preparation, not plugin inference: production `GFX` inference may do CPU compile/planning/setup work, input upload, and final output readback, but intermediate tensor execution must remain on GPU without CPU fallback or hidden CPU copies. Split-like producers are materialized as host logical views/copies using the static model output contract when available; static `ShapeOf` and 1D shape-list `Gather` producers can also be resolved in the runner for recursive diagnostics. This keeps macOS Metal, Android Vulkan, and RPi Vulkan on the same shape contract instead of diverging on diagnostic zero-length branches or remote shader-`Int64` lowering limits. Set `--per-op-recursive-limit` and `--per-op-recursive-trace` only when you deliberately want a bounded remote diagnostic that reports the producer where materialization stopped instead of running the whole upstream chain.
- Targeted per-op `compile_skip` and `infer_skip` are failures, not successful matches: the runner now reports `PER_OP_SKIPPED` and returns non-zero instead of hiding the gap behind `PER_OP_MATCH`. `bench/gfx_eval.py` accepts that specific compare-runner exit code and writes it to JSON as `mode=per_op_skipped` for host, Android, and SSH runs.
- SSH runs through `bench/gfx_eval.py --ssh-device-file` accept connection settings from a local device configuration file. Keep those files outside the published module tree, and keep the SSH layer as launch plumbing only; it must not introduce target-specific accuracy skips or runtime fallback behavior.
- `VariadicSplit` regressions should include an inferred `-1` length case so backend-local split code cannot drift from the shared runtime split plan.
- `ov_gfx_compare_runner` also supports boolean tensors, `--single-op-output`, `--tinyllama-prompt-inputs`, `--input-image`, and an extra `Select` mismatch probe for harder data-dependent failures
- `ov_gfx_compare_runner` now prints outputs as `friendly_name:port` and reports `max_index`, reference value, and GFX value for the worst mismatch
- `ov_gfx_conv_shape_bench` is useful when stage-policy, Metal placement, Vulkan Conv2D retry behavior, or Broadcom V3D dispatch tiling changes need a quick before/after sample on fixed Conv2D shapes without running a full benchmark flow. Keep GFX runs on `--device GFX`; run `CPU` only as a separate comparator when documenting performance corridors. For RPi5 pointwise Conv2D checks, `--case yolo26x_pw_48_48_160` should show the shared MLIR/SPIR-V dispatch using the occupancy-aware V3D policy after `OV_GFX_DEBUG=1`, not a full 256-thread tile.
- keep `ov_gfx_compare_runner` accuracy-only and use `benchmark_app` for perf
- use `ov_gfx_microbench` for `MB0` to `MB3`, calibration artifacts, and profiling triage rather than for acceptance perf numbers
- for the full profiling workflow and external tracing commands, use `PROFILING_RUNBOOK.md`
- for the microbench JSON and calibration-artifact contract, use `MICROBENCH_SCHEMA.md`
- for optional automation around those flows, use `tools/gfx_profile_runbook.py`, `tools/gfx_microbench_smoke.py`, `tools/gfx_calibration_diff.py`, and `tools/gfx_external_trace_summary.py`
- Reuse-related regressions are often easier to catch with focused unit tests than with full end-to-end backend suites

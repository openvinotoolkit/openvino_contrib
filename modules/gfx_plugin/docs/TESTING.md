# Testing Guide

This guide summarizes validation for `modules/gfx_plugin`.

## Main Targets

`tests/CMakeLists.txt` defines the primary test binaries:

- `ov_gfx_func_tests`: plugin-facing behavior and OpenVINO shared-test coverage
- `ov_gfx_unit_tests`: focused compiler, manifest, runtime, MLIR, cache,
  property, profiling, and backend tests
- `ov_gfx_runtime_micro_tests`: small runtime-subgraph regression checks
- `ov_gfx_compare_runner`: accuracy and per-op diff tool
- `ov_gfx_microbench`: MB0-MB3 microbench and calibration workflow
- `ov_gfx_conv_shape_bench`: representative Conv2D compile-plus-infer probe

Build:

```bash
cmake --build build-gfx-plugin --target ov_gfx_func_tests
cmake --build build-gfx-plugin --target ov_gfx_unit_tests ov_gfx_runtime_micro_tests
cmake --build build-gfx-plugin --target ov_gfx_compare_runner ov_gfx_microbench ov_gfx_conv_shape_bench
```

Run all GFX-labeled tests:

```bash
ctest --test-dir build-gfx-plugin --output-on-failure -L GFX
```

Run a focused gtest filter:

```bash
find build-gfx-plugin -name ov_gfx_unit_tests -type f
DYLD_LIBRARY_PATH=/path/to/openvino/runtime/libs \
  <path-to-ov_gfx_unit_tests> --gtest_filter=GfxStagePolicy.*
```

## Test Layout

- `tests/unit/`: focused unit tests for plugin logic, MLIR lowering, runtime
  helpers, caches, submission, profiling, backend registry contracts, and
  OpenCL source artifacts
- `tests/backends/metal/`: Metal-specific runtime, MPSRT, memory, and behavior
  coverage
- `tests/integration/`: plugin integration checks
- `tests/shared_tests_instances/`: OpenVINO shared-test wiring
- `tests/tools/`: compare runner, microbench, Conv shape bench, and standalone
  OpenCL experiment tools
- `bench/`: optional evaluation orchestration helpers
- `tools/`: profiling and microbench post-processing helpers

Only Metal has backend-specific runtime tests in the current source tree; OpenCL
runtime behavior is covered through source-artifact unit tests, integration
paths, and target execution.

## What To Test

Add or update tests when changing:

- public properties or backend selection
- `query_model()` or support probing
- compiler backend registry, operation policies, lowering plans, manifests,
  executable bundles, runtime executable descriptors, or artifact payloads
- MLIR builders, passes, source plans, or runtime-value planning
- stage policy, placement, fusion, precision, or submit policy
- OpenCL source-artifact metadata, source coverage, chunking, constant
  materialization, boolean buffer handling, or dynamic shape scalars
- Metal MPS/MPSGraph placement, MPSRT records, storage bridges, MSL binding
  plans, or request-time resource binding
- stateful `ReadValue` / `Assign`
- reusable infer plans, reusable host outputs, immutable const caches, prepared
  binding caches, or workspace allocation
- output-source tracking, output aliases, or view-style Split/VariadicSplit
  behavior
- profiling JSON fields, stage estimates, target-profile counters, or
  microbench schema
- compare-runner behavior, precision-aware tolerances, or golden-reference flow

## Recommended Test Selection

For MLIR or transform changes:

- prefer `tests/unit/mlir_*_test.cpp`
- add IR-shape assertions where possible
- use runtime tests only after the lowering contract is covered

For plugin/property changes:

- update `tests/unit/plugin_tests.cpp`
- check `ov::available_devices`, `ov::device::id`, `ov::supported_properties`,
  `GFX_BACKEND`, and compiled-model property behavior

For compiler-service, manifest, or executable-descriptor changes:

- `tests/unit/gpu_backend_base_test.cpp`
- `tests/unit/gfx_backend_architecture_contract_test.cpp` when backend target
  identity, kernel-unit registration, or manifest contracts move
- `tests/unit/plugin_tests.cpp` when `query_model()` or compile behavior moves
- backend artifact tests when payload materialization reaches Metal or OpenCL
  runtime loaders
- Metal vendor-descriptor coverage when MPS/MPSGraph payloads reach
  `MpsrtVendorPrimitiveStage`

For scheduling, cache, or infer-path changes:

- `tests/unit/gfx_stage_policy_test.cpp`
- `tests/unit/gfx_parallelism_test.cpp`
- `tests/unit/infer_submission_test.cpp`
- `tests/unit/infer_pipeline_reuse_test.cpp`
- `tests/unit/gpu_const_cache_test.cpp`
- `tests/unit/kernel_arg_reuse_test.cpp`
- `tests/unit/gpu_backend_base_test.cpp`
- `ov_gfx_runtime_micro_tests` focused files such as
  `tests/unit/gfx_activation_runtime_test.cpp`,
  `tests/unit/gfx_add_runtime_test.cpp`,
  `tests/unit/gfx_matmul_runtime_test.cpp`,
  `tests/unit/gfx_multiply_runtime_test.cpp`,
  `tests/unit/gfx_reduce_logical_runtime_test.cpp`,
  `tests/unit/gfx_split_runtime_test.cpp`
- `tests/unit/gfx_softmax_kernel_contract_test.cpp` and
  `tests/unit/gfx_pool_kernel_contract_test.cpp` for Softmax and Pooling
  lowering, source-artifact, kernel-registry, and payload contracts

For OpenCL source-artifact changes:

- start with `tests/unit/gfx_opencl_source_artifacts_test.cpp`
- include `tests/unit/gfx_activation_kernel_contract_test.cpp`,
  `tests/unit/gfx_eltwise_kernel_contract_test.cpp`, or
  `tests/unit/gfx_matmul_kernel_contract_test.cpp` when the generated source
  unit contract for those families changes
- include `tests/unit/gfx_reduction_kernel_contract_test.cpp` when generated
  f32 reduction, logical-bool reduction, reduction MLIR lowering, or backend
  reduction kernel-unit routing changes
- include `tests/unit/gfx_softmax_kernel_contract_test.cpp` when generated
  Metal Softmax/LogSoftmax payloads, OpenCL static or dynamic-static-rank
  Softmax artifacts, axis metadata, or Softmax kernel-unit routing changes
- include `tests/unit/gfx_pool_kernel_contract_test.cpp` when OpenCL generated
  Pool2D artifacts, Metal MPS Pool2D vendor routing, or Pooling kernel-unit
  registration changes
- use `tests/unit/gfx_opencl_source_artifact_verifier.hpp` for reusable
  OpenCL source-artifact assertions instead of duplicating role/scalar checks
- keep reusable generated activation and elementwise case data in
  `tests/unit/gfx_activation_contract_cases.hpp`,
  `tests/unit/gfx_activation_opencl_contract_cases.cpp`,
  `tests/unit/gfx_activation_msl_contract_cases.cpp`,
  `tests/unit/gfx_eltwise_contract_cases.hpp`, and
  `tests/unit/gfx_eltwise_opencl_contract_cases.cpp`
- include `tests/unit/gfx_eltwise_opencl_source_artifacts_test.cpp` when
  elementwise OpenCL artifact metadata or source identity changes
- use `tests/unit/gpu_backend_base_test.cpp` when a source artifact is expected
  to appear in the compiler executable bundle
- add runtime coverage when dynamic OpenCL loading, buffer binding, runtime
  output shape, static f32 scalar binding, constant materialization, boolean
  storage, or command execution changes
- use `tests/tools/ov_gfx_opencl_conv_microbench.py` and
  `tests/tools/ov_gfx_opencl_conv_microbench_android.cpp` only as kernel-family
  experiments before promotion into the plugin contract

For Metal placement or MPSRT changes:

- cover manifest/source-plan records in `tests/unit/gfx_stage_policy_test.cpp`,
  `tests/unit/basic_ops_internal_test.cpp`, or
  `tests/unit/gpu_backend_base_test.cpp`
- cover compiler-owned generated MSL and MPS/MPSGraph `VendorDescriptor`
  payloads in `tests/unit/gpu_backend_base_test.cpp`
- cover generated reduction MSL source plans and payload routing in
  `tests/unit/gfx_reduction_kernel_contract_test.cpp`
- cover generated Softmax/LogSoftmax MSL source plans and payload routing in
  `tests/unit/gfx_softmax_kernel_contract_test.cpp`
- cover Pool2D vendor-route and MSL-fallback rejection contracts in
  `tests/unit/gfx_pool_kernel_contract_test.cpp`
- cover request-time execution in `tests/backends/metal/gpu_backend_test.mm`
  when the route reaches encode time
- use `tests/unit/memory_device_integration_test.mm` for Metal memory/device
  integration behavior

## Compare Runner

Use `ov_gfx_compare_runner` for correctness and narrowing:

- full-model reference comparison
- `--per-op-all`
- `--single-op-output`
- real-image PPM inputs
- golden-reference output directories
- `--gfx-inference-precision f16|f32`
- boolean-output and Select mismatch probes
- detailed first/worst mismatch output

Keep it accuracy-focused. Use `benchmark_app` or `ov_gfx_microbench` for
performance data.

## Microbench And Profiling

Use `ov_gfx_microbench` for MB0-MB3 overhead and calibration triage. The
supported backend argument is:

```bash
./ov_gfx_microbench --backend auto|metal|opencl --warmup 1 --iterations 3
```

Use:

- `docs/MICROBENCH_SCHEMA.md` for JSON and calibration artifact fields
- `docs/PROFILING_RUNBOOK.md` for platform profiling commands
- `tools/gfx_profile_runbook.py` to print ready-to-run command sets
- `tools/gfx_microbench_smoke.py` for artifact round-trip checks
- `tools/gfx_calibration_diff.py` for calibration comparison
- `tools/gfx_external_trace_summary.py` for trace post-processing

## Cross-Device Strategy

Treat backend routes separately:

- macOS validates the Metal/MPS/MPSRT/MSL path.
- Android or Linux targets with a working OpenCL GPU runtime validate the
  OpenCL source-kernel path.
- Raspberry Pi validation should use the OpenCL runtime that belongs to the
  target deployment contract.

A Metal pass does not prove OpenCL correctness. An OpenCL failure should be
fixed in the shared manifest/lowering/artifact path first unless the evidence
points to a concrete backend runtime boundary.

Do not turn device failures into skips until the same op has been checked
through the narrow unit path and a real backend execution path.

## Practical Pre-Commit Checks

For docs-only changes:

```bash
git diff --check
rg -n "sensitive-placeholder" README.md docs .codex/skills || true
```

For documentation/security publication tasks, skip build and test targets
unless they are explicitly requested. Use source inspection, security grep,
stale-reference grep, `git diff --check`, and staged diff review instead.

For source or build-file changes:

```bash
cmake --build build-gfx-plugin --target openvino_gfx_plugin ov_gfx_unit_tests
ctest --test-dir build-gfx-plugin --output-on-failure -L GFX
```

Use narrower gtest filters first when a full label run is not needed or would be
too heavy for the current change.

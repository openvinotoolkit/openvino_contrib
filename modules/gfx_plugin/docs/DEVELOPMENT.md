# Development Guide

This guide is for contributors working inside `modules/gfx_plugin`.

## Prerequisites

- CMake 3.13 or newer
- Ninja recommended
- OpenVINO Developer Package
- Metal SDK/frameworks on macOS for the Metal backend
- OpenCL runtime on non-Apple targets for the OpenCL source-kernel backend
- Optional `third_party/clvk` and `third_party/clspv` submodules when building
  the Raspberry/Linux OpenCL bundle

The module vendors LLVM/MLIR in `third_party/llvm-project` and builds the
required MLIR components as part of the CMake flow. The CLVK and CLSPV
directories are submodules used only by the optional Raspberry/OpenCL bundle.
Do not modify vendored LLVM or third-party submodule contents unless a task
explicitly requires it.

## Configure And Build

Configure from the module root:

```bash
cmake -S . -B build-gfx-plugin -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DOpenVINODeveloperPackage_DIR=/path/to/openvino/install/cmake \
  -DENABLE_TESTS=ON \
  -DGFX_DEFAULT_BACKEND=auto
```

Build common targets:

```bash
cmake --build build-gfx-plugin --target openvino_gfx_plugin
cmake --build build-gfx-plugin --target ov_gfx_func_tests ov_gfx_unit_tests ov_gfx_runtime_micro_tests
cmake --build build-gfx-plugin --target ov_gfx_compare_runner ov_gfx_microbench
```

Useful CMake options:

- `GFX_ENABLE_METAL`
- `GFX_ENABLE_OPENCL`
- `GFX_ENABLE_RASPBERRY_OPENCL_TOOLCHAIN`
- `GFX_RASPBERRY_CLVK_SOURCE_DIR`
- `GFX_RASPBERRY_CLSPV_SOURCE_DIR`
- `GFX_RASPBERRY_OPENCL_BUILD_DIR`
- `GFX_RASPBERRY_OPENCL_BUNDLE_DIR`
- `GFX_DEFAULT_BACKEND=auto|metal|opencl`
- `ENABLE_TESTS`

On macOS, `cmake/GfxBackendConfig.cmake` disables OpenCL and resolves the Apple
route to Metal/MPS/MPSRT/MSL. On non-Apple builds, the OpenCL source backend is
the runtime route when it is enabled in the build.

Build-system notes:

- `cmake/GfxSources.cmake` is the source list used by both the module root and
  `src/CMakeLists.txt`.
- `src/CMakeLists.txt` embeds selected `.cl` and `.metal` helper sources through
  `cmake/KernelSource.hpp.in`; generated headers stay in the build tree and
  must not be committed.
- `gfx_plugin_core`, `gfx_runtime_common`, and `gfx_runtime_mlir` contain shared
  code.
- `gfx_plugin_metal` / `gfx_runtime_metal` and
  `gfx_plugin_opencl` / `gfx_runtime_opencl` contain backend-specific code.
- `src/plugin/gfx_backend_config.hpp.in` is configured into the build tree with
  backend availability booleans and the resolved default backend.
- The OpenCL backend dynamically loads the target OpenCL runtime; it does not
  require a compile-time OpenCL SDK link.
- `cmake/GfxRaspberryOpenCLToolchain.cmake` can build and stage a plugin-local
  CLVK/CLSPV OpenCL bundle on Linux ARM targets when OpenCL support is
  available. It expects initialized `third_party/clvk`, `third_party/clspv`,
  and CLVK dependency submodules, plus host LLVM tools from the target
  toolchain layout.
- `cmake/InstallRaspberryOpenCLBundle.cmake` stages `libOpenCL.so.0.1`,
  `clspv`, optional `llvm-spirv`, and local `libOpenCL.so*` symlinks into
  `GFX_RASPBERRY_OPENCL_BUNDLE_DIR`.
- `cmake/WriteGfxTestPluginsXml.cmake` writes the controlled test
  `plugins.xml` used by GFX test binaries.
- Android and generic cross builds forward toolchain settings into the vendored
  LLVM/MLIR configure step.
- The build treats warnings as errors through `-Werror` on Clang/GCC and `/WX`
  on MSVC.

## Where To Start Reading

Read in this order:

1. `README.md`
2. `docs/ARCHITECTURE.md`
3. `src/plugin/plugin.cpp`
4. `src/compiler/gfx_compiler_service.*`
5. `src/compiler/backend_registry.*`
6. `src/compiler/manifest.*` and `src/compiler/executable_bundle.*`
7. `src/plugin/compiled_model.cpp`
8. `src/plugin/infer_pipeline.*`
9. `src/plugin/infer_submission.*`
10. the backend directory you are changing

For runtime planning, also inspect:

- `src/compiler/lowering_planner.*`
- `src/compiler/operation_support.*`
- `src/compiler/kernel_registry.*`
- `src/runtime/gfx_stage_policy.*`
- `src/runtime/gfx_parallelism.*`
- `src/runtime/gfx_partitioning.*`
- `src/runtime/executable_descriptor.*`
- `src/runtime/view_only_stage.*`
- `src/runtime/gfx_target_profile.*`
- `src/runtime/gfx_profiling_trace_sink.*`
- `src/kernel_ir/gfx_kernel_manifest.hpp`
- `src/kernel_ir/gfx_custom_kernel_families.*`
- `src/kernel_ir/gfx_kernel_source.*`
- `src/kernel_ir/gfx_opencl_source_artifacts.*`
- `src/mlir/gfx_stage_runtime_values.*`
- `src/mlir/gfx_backend_custom_kernel_adapter.*`

For Metal placement, MPSRT, or MSL source planning, also inspect:

- `src/backends/metal/compiler/`
- `src/backends/metal/runtime/metal_runtime_kernel_loader.*`
- `src/backends/metal/runtime/mpsrt_vendor_primitive_stage.*`
- `src/runtime/gfx_mpsrt_abi.hpp`
- `src/runtime/gfx_mpsrt_model.*`
- `src/runtime/gfx_mpsrt_plan.hpp`
- `src/runtime/gfx_mpsrt_program.hpp`
- `src/runtime/gfx_mpsrt_kernel_manifest_adapter.hpp`
- `src/mlir/gfx_apple_stage_pipeline.*`
- `src/mlir/gfx_apple_vendor_descriptors.*`
- `src/mlir/gfx_mpsrt_dialect.*`
- `src/mlir/gfx_mpsrt_ops.*`
- `src/mlir/gfx_mpsrt_source_plan.hpp`
- `src/mlir/msl_codegen_apple_msl*.{cpp,hpp}`
- `src/mlir/msl_codegen_apple_mps.*`
- `src/mlir/msl_codegen_matmul_*`
- `src/mlir/msl_codegen_attention.*`
- `src/mlir/msl_codegen_compressed_matmul.*`
- `src/backends/metal/runtime/mpsrt/`

For OpenCL source execution, start with:

- `src/backends/opencl/compiler/`
- `src/backends/opencl/plugin/`
- `src/backends/opencl/runtime/opencl_api.*`
- `src/backends/opencl/runtime/opencl_buffer_manager.*`
- `src/backends/opencl/runtime/opencl_program_cache.*`
- `src/backends/opencl/runtime/opencl_runtime_kernel_loader.*`
- `src/backends/opencl/runtime/opencl_source_stage.*`
- `src/backends/opencl/runtime/stage_factory.*`
- `tests/unit/gfx_opencl_source_artifacts_test.cpp`
- `tests/unit/gfx_opencl_source_artifact_verifier.hpp`
- `tests/unit/gfx_activation_contract_cases.hpp`
- `tests/unit/gfx_activation_opencl_contract_cases.cpp`
- `tests/unit/gfx_eltwise_contract_cases.hpp`
- `tests/unit/gfx_eltwise_opencl_contract_cases.cpp`
- `tests/unit/gfx_reduction_kernel_contract_test.cpp`

## Adding Or Changing An Operation

1. Update support probing in `src/mlir/`.
2. Update compiler route selection when support changes:
   - common compiler contracts in `src/compiler/`
   - Metal policy, kernel registry, or artifact resolver in
     `src/backends/metal/compiler/`
   - OpenCL policy or kernel registry in `src/backends/opencl/compiler/`
3. Add or adjust lowering in the relevant `mlir_builder_*.cpp`,
   source-plan helper, or transform.
4. Decide the shared runtime contract before adding backend code:
   - compiler manifest/executable descriptor for stage ABI and artifact payloads
   - stage policy for placement/fusion/submission
   - kernel manifest for custom-kernel ABI
   - runtime-value payloads for dynamic metadata
   - OpenCL source artifact for source-kernel execution
   - MPSRT/Apple MSL source plan for Metal execution
5. Add backend code only where the shared contract crosses into Metal or OpenCL.
   Backend stage creation now requires the matching runtime executable
   descriptor for executable routes; do not reconstruct kernel or vendor
   payloads from the OpenVINO node in request-time code.
6. Add focused unit tests first.
7. Add backend or functional tests when behavior is externally visible.
8. Update docs when public properties, supported shapes, route selection,
   profiling, or test workflow changes.

Common operation families that need extra care:

- dynamic-shape data movement: `ShapeOf`, `Concat`, `Broadcast`, `Select`,
  `Slice`, `Range`, and `Tile`
- stateful `ReadValue` / `Assign`
- view-style `Split` / `VariadicSplit` aliases
- Metal MPS/MPSGraph vendor stages and MPSRT storage bridges
- Metal custom MSL source plans with explicit kernel-buffer order
- OpenCL source artifacts with scalar ABI, static u32/f32 scalars, constants,
  chunking, and boolean output padding
- OpenCL generated kernel units such as activation, elementwise, f32 MatMul,
  bounded f32/f16 Interpolate, f32 reduction, boolean reduction, f32/f16
  Softmax, dynamic-static-rank f32/f16 Softmax, f32/f16 Pool2D, ShapeOf, Tile,
  compare/select, logical-bool elementwise, and generated Concat/Split
- OpenCL reduction routes, where numeric f32 reductions and logical boolean
  reductions use separate generated source ids but the same static axis
  metadata contract
- generated activation `Swish` routes, where default/static beta and runtime
  scalar beta must keep the MLIR, Metal MSL, and OpenCL artifact contracts
  aligned
- generated Softmax routes, where Metal `Softmax`/`LogSoftmax` and OpenCL
  `Softmax` must keep axis normalization, scalar metadata, and kernel-unit ids
  aligned with their backend-specific contracts
- Pooling routes, where OpenCL generated Pool2D covers static f32/f16 NCHW
  window contracts and Metal Pool2D must use a valid MPS-family vendor route
- LLM-oriented fusions such as `RoPE`, compressed `MatMul`, and SDPA variants

## Shared Versus Backend-Specific Code

Prefer these shared locations:

- backend target, operation legality, route selection, manifest, executable
  bundles, and artifact descriptors: `src/compiler/`
- graph rewrites: `src/transforms/`
- support probing and lowering: `src/mlir/`
- stage policy and parallelism: `src/runtime/gfx_stage_policy.*`,
  `gfx_parallelism.*`, and `gfx_partitioning.*`
- binding and manifest contracts: `src/kernel_ir/` and
  `src/mlir/gfx_backend_custom_kernel_adapter.*`
- infer planning and submission: `src/plugin/infer_pipeline.*` and
  `src/plugin/infer_submission.*`

Use backend directories only for real backend boundaries:

- Metal operation support policy, Metal kernel registry, and generated MSL
  payload materialization belong under `src/backends/metal/compiler/`.
- Metal Objective-C++ APIs, MTL resources, MPS/MPSGraph setup, MSL compilation,
  and command encoding belong under `src/backends/metal/`.
- OpenCL operation support policy and OpenCL kernel registry belong under
  `src/backends/opencl/compiler/`.
- OpenCL platform/device discovery, program compilation, buffer management, and
  kernel enqueue code belong under `src/backends/opencl/`.

Do not duplicate shared ABI, route, or shape rules in backend request code.

## OpenCL Source Artifacts

`src/kernel_ir/gfx_opencl_source_artifacts.*` is the source of truth for:

- source id and entry point
- tensor role order
- scalar ABI
- source-static u32/f32 scalar values
- local size
- element-count source
- dynamic-shape scalar metadata
- direct constant materialization
- boolean buffer padding
- generated Concat/Split chunk artifacts

`src/backends/opencl/runtime/opencl_source_stage.*` should stay a generic
artifact executor. Add metadata to artifacts rather than adding op-specific
branches to the executor.

The current handwritten OpenCL source exception is `opencl/baseline/transpose_f32`.
Do not introduce a new baseline exception unless the generated-source contract
cannot express the route and the exception is documented, tested, and reviewed.

Generated or embedded source payloads should flow through compiler artifact
descriptors and runtime kernel loaders. Do not pass ad-hoc source strings
through plugin or infer-request properties.

When adding an embedded OpenCL source unit:

- place the `.cl`, `.cpp`, and `.hpp` wrapper under
  `src/kernel_ir/opencl_kernels/`
- add the source to `src/CMakeLists.txt` through `gfx_embed_kernel_source()`
- add the wrapper source/header to `cmake/GfxSources.cmake`
- route it from `gfx_opencl_source_artifacts.*` with explicit source id,
  entry point, route kind, scalar ABI, and shape/type limitations
- cover source identity, scalar metadata, support probing, and payload routing
  in `tests/unit/gfx_opencl_source_artifacts_test.cpp`,
  family-specific contract case files, and
  `tests/unit/gpu_backend_base_test.cpp` when the compiler bundle is affected
- update `tests/unit/gfx_backend_architecture_contract_test.cpp` when kernel
  registry, backend-target identity, or manifest-routing contracts change

For generated activation changes, update
`tests/unit/gfx_activation_contract_cases.hpp`,
`tests/unit/gfx_activation_opencl_contract_cases.cpp`, and
`tests/unit/gfx_activation_msl_contract_cases.cpp` together. For generated
elementwise OpenCL changes, update `tests/unit/gfx_eltwise_contract_cases.hpp`,
`tests/unit/gfx_eltwise_opencl_contract_cases.cpp`, and
`tests/unit/gfx_eltwise_opencl_source_artifacts_test.cpp`.

For reduction source-unit changes, update
`tests/unit/gfx_reduction_kernel_contract_test.cpp` and the shared
`tests/unit/gfx_opencl_source_artifact_verifier.hpp` helper. Keep numeric f32
and logical boolean generated reduction source ids, entry points, static u32
metadata, kernel registry entries, and Metal/OpenCL artifact payloads aligned.

For Softmax source-unit changes, update
`tests/unit/gfx_softmax_kernel_contract_test.cpp` and the shared
`tests/unit/gfx_opencl_source_artifact_verifier.hpp` helper. Keep Metal
f32/f16 Softmax and LogSoftmax source ids, OpenCL static and dynamic-static-rank
Softmax source ids, runtime-parameter roles, scalar metadata, and kernel
registry entries aligned.

For Pool2D source-unit changes, update
`tests/unit/gfx_pool_kernel_contract_test.cpp`,
`tests/integration/gfx_pooling_func_test.cpp` when externally visible behavior
changes, and `tests/unit/gfx_backend_architecture_contract_test.cpp` when
kernel-unit registration changes. Do not add a Metal MSL Pool2D fallback unless
there is explicit MPS-family rejection evidence and an op-owned narrow artifact
contract.

## Metal MPSRT And MSL

Metal placement must stay coordinated across:

- `gfx_stage_policy.*`
- `GfxKernelStageManifest`
- custom-kernel family metadata
- Apple source-plan helpers
- typed `GfxMpsrtProgram` / generated `gfx_mpsrt_ops`
- `GfxMpsrtBuilderPlan`
- `src/runtime/gfx_mpsrt_model.*`
- `src/backends/metal/runtime/mpsrt/`

When a manifest supplies explicit external-buffer roles, those roles define the
runtime ABI. MSL buffer scans and signature hints are diagnostics/fallback
inputs only; they must not widen or shrink a typed MPSRT binding contract.

Embedded MPSRT helper kernels live under `src/kernel_ir/metal_kernels/` and are
exposed through runtime loaders. Keep helper source ownership there instead of
reintroducing large inline MSL strings in request encoders.

Compiler-owned Metal payloads now include both generated MSL sources and
MPS/MPSGraph `VendorDescriptor` payloads. When adding a vendor primitive route,
update the Metal operation policy, `metal_kernel_artifacts.*`, the typed vendor
descriptor helpers in `src/mlir/gfx_apple_vendor_descriptors.*`, and
`mpsrt_vendor_primitive_stage.*` only if the existing runtime contract cannot
express the new primitive. Do not rebuild vendor descriptors from request-time
node checks.

Generated Metal activation, elementwise, reduction, and Softmax paths are
planned through `src/mlir/msl_codegen_apple_msl_activation.*`,
`src/mlir/msl_codegen_apple_msl_eltwise.*`,
`src/mlir/msl_codegen_apple_msl_reduction.*`, and
`src/mlir/msl_codegen_apple_msl_softmax.*`. Keep those source plans aligned
with `src/backends/metal/compiler/metal_kernel_registry.cpp`,
`metal_kernel_artifacts.cpp`, and embedded helper source wrappers under
`src/kernel_ir/metal_kernels/`. For `Swish`, keep static-beta and runtime-beta
binding roles aligned with `src/mlir/mlir_builder_unary.cpp` and the OpenCL
source artifact ABI. For Softmax, keep generated `Softmax` and `LogSoftmax`
runtime-parameter roles aligned with the registered f32/f16 kernel units.

## Properties

Public GFX properties are declared in
`include/openvino/gfx_plugin/properties.hpp` and exposed through
`src/plugin/gfx_property_lists.cpp`.

Before changing property behavior, check:

- `src/plugin/gfx_property_utils.*`
- `src/plugin/plugin.cpp`
- `src/plugin/compiled_model.cpp`
- `tests/unit/plugin_tests.cpp`
- `docs/USAGE.md`

## Testing Expectations

Use the narrowest relevant checks first:

```bash
cmake --build build-gfx-plugin --target ov_gfx_unit_tests
find build-gfx-plugin -name ov_gfx_unit_tests -type f
DYLD_LIBRARY_PATH=/path/to/openvino/runtime/libs \
  <path-to-ov_gfx_unit_tests> --gtest_filter=GfxStagePolicy.*
```

Before ordinary source commits, run at least:

```bash
git diff --check
ctest --test-dir build-gfx-plugin --output-on-failure -L GFX
```

For documentation/security publication tasks, do not run build or test targets
unless explicitly requested. Use source inspection, targeted security/stale
reference grep, `git diff --check`, and staged diff review for that gate.
For compiler, manifest, or executable-descriptor changes, include
`GpuBackendBaseTest.*` or a narrower relevant filter from
`tests/unit/gpu_backend_base_test.cpp`.

For backend registry or generated source-unit changes, also include the focused
contract suites in `tests/unit/gfx_backend_architecture_contract_test.cpp`,
`tests/unit/gfx_activation_kernel_contract_test.cpp`,
`tests/unit/gfx_eltwise_kernel_contract_test.cpp`, and
`tests/unit/gfx_matmul_kernel_contract_test.cpp` as applicable.

## Public Repository Hygiene

Keep these out of public commits:

- `AGENTS.md` and local agent notes
- build directories and generated reports
- `__pycache__`, `.DS_Store`, IDE metadata, temporary logs, and local profiles
- sensitive access material, machine names, local absolute paths, and device configs
- stale architecture notes for removed routes

Do not use `git add .`. Stage only reviewed files.

## Files To Avoid Without Explicit Reason

- `third_party/llvm-project/`
- `third_party/clvk/` and `third_party/clspv/` contents; stage them as
  submodule gitlinks only when the submodule pointers intentionally change
- local ignored artifacts
- root-level technical notes outside this module
- backend source files unrelated to the requested behavior
- generated build-tree files

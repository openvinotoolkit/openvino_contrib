# OpenVINO GFX Plugin

Experimental OpenVINO device plugin registered as `"GFX"`.

This directory is intended to be self-contained. Start here, then read:
- `docs/ARCHITECTURE.md` for the current code-truth architecture
- `docs/DEVELOPMENT.md` for build, extension, and debugging guidance
- `docs/TESTING.md` for test targets and workflows
- `docs/USAGE.md` for integration examples

## What The Plugin Does
GFX is an OpenVINO runtime plugin that compiles an `ov::Model` into a backend-specific GPU execution pipeline.
The current direction is mobile-class GPU execution rather than large datacenter GPUs:
- Metal on macOS
- OpenCL source kernels on non-Apple builds when the dynamic OpenCL runtime is available
- Vulkan as a legacy diagnostic backend for SPIR-V/Vulkan investigations

The codebase uses a shared frontend and a stage-based runtime:
1. OpenVINO graph transformations run in `src/transforms/`
2. Backend-aware support probing is driven by MLIR builders in `src/mlir/`
3. `CompiledModel` builds a pipeline of `GpuStage` objects
4. Backend-specific infer requests bind tensors and execute the stage pipeline

Recent runtime work extends this model in two directions:
- compile-time stage planning now picks layout, fusion, and execution policy per stage
- backend runtimes can choose specialized direct, source-artifact, or chunked execution routes for selected ops while Conv2D and GroupConv stay on shared custom-kernel contracts
- infer execution can batch stage recording into submission windows and reuse prepared bindings or immutable device buffers across requests
- infer submission can keep a direct producer-consumer chain in one command-buffer window even after a soft budget boundary, while still stopping at layout, split, transpose, softmax, and attention boundaries
- Metal infer execution can now reuse one compute encoder across consecutive dispatches and skip redundant pipeline or buffer rebinds when the command-buffer state is unchanged
- device-aware scheduling now uses backend-reported execution limits and device-family classification through shared `gfx_parallelism.*` and `gfx_partitioning.*` helpers
- Vulkan Conv2D dispatch planning can block several output channels per work item through the same `gfx_parallelism.*` plan and `gfx.dispatch_channel_block` MLIR/SPIR-V metadata used by the shared convolution lowering path
- Metal stage planning now also chooses an internal placement domain per stage: Apple MPS/MPSGraph vendor primitives for selected ops, or Apple MSL buffer dispatch for custom-kernel paths
- `GFX_DIAGNOSTIC_F32_MPS_IMAGE` and `ov_gfx_compare_runner --diagnostic-f32-mps-image` keep the selected `f32` Conv/GroupConv/Pool image route available as a diagnostic localization switch through the same planner/MPSRT route; production placement is still owned by normal stage policy and quality gates
- the Metal compile path can serialize that placement into a backend-neutral MPSRT runtime model under `src/runtime/gfx_mpsrt_model.*` with explicit tensor descriptors, external-buffer roles, a typed `GfxMpsrtProgram` facade, generated `gfx_mpsrt_ops` materialization, and explicit storage-bridge descriptors before request-time execution
- that MPSRT path is no longer limited to one annotated MSL dispatch; it can now carry vendor-only and hybrid multi-stage plans covering Apple MPS/MPSGraph GEMM, Conv2D, GroupConv, Pool2D, Resize2D, Softmax, TopK, and SDPA stages together with custom MSL epilogues or dispatch stages
- Metal MSL source planning is now part of the shared MLIR/runtime layer and is split by responsibility into Apple MSL custom-kernel helpers, Apple MPS vendor-source-plan helpers, MatMul-specific Metal/MPSRT helpers, an OpenCL source-artifact path, and a SPIR-V binding adapter for the Vulkan side
- OpenCL source-kernel execution now has its own backend plugin/runtime layer, dynamic OpenCL API loader, source-artifact manifest path, program cache, buffer manager, baseline f32 data-movement kernels, typed f32/i32/i64 convert casts, and elementwise kernels
- infer requests can also keep per-request stateful variable buffers for `ReadValue` / `Assign` style subgraphs instead of treating them as ordinary stateless stage edges
- output allocation can now reuse workspace-managed intermediate slots across stages based on liveness instead of always keeping one dedicated buffer per stage output
- shared prepared-binding caches can now grow beyond their initial capacity when a workload introduces many distinct compatible binding tables
- infer-time stage profiling can now attach lightweight `bytes_in`, `bytes_out`, `macs_est`, and `flops_est` estimates to `stage_execute` segments
- backend buffer managers now report a shared target profile so profiling can show the resolved backend, device family, workgroup limits, memory alignment, and kernel capability flags through one JSON surface

This is not the old monolithic `MlirBackend` architecture that earlier design notes experimented with.

## Current Status
- Device name: `GFX`
- Backends: `metal`, `opencl`, `vulkan`
- Target class: mobile and embedded GPUs
- Expected integration: recent OpenVINO Developer Package builds
- Tested on:
  - Apple M1 Pro via Metal
  - Samsung Galaxy S25 through OpenCL source-kernel validation and legacy Vulkan diagnostics
  - Raspberry Pi 5B through OpenCL source-kernel validation and legacy Vulkan diagnostics
- OpenCL is the preferred non-Apple default when enabled and available; Vulkan remains available for legacy SPIR-V diagnostics
- No partial CPU fallback: unsupported models fail during `compile_model()`
- `query_model()` is backend-aware and follows the same support probing path as compilation
- `import_model()` reloads an OpenVINO model and recompiles it
- `export_model()` serializes the OpenVINO model, not a backend pipeline cache
- `GfxRemoteContext` and `GfxRemoteTensor` are implemented, but capabilities still depend on the compiled backend

## Source Layout
- `include/openvino/gfx_plugin/`: public plugin headers
- `src/plugin/`: `Plugin`, `CompiledModel`, shared property handling, pipeline construction
- `src/runtime/`: backend-neutral runtime abstractions and helpers
- `src/runtime/gfx_mpsrt_*`: shared ABI, stage-plan, builder-plan, runtime-model, program, storage-bridge, and manifest-adapter helpers for the Apple MPS/MSL split
- `src/kernel_ir/gfx_kernel_manifest.hpp`: backend-neutral manifest for vendor-primitive versus custom-kernel stage contracts
- `src/kernel_ir/gfx_custom_kernel_families.*`: custom-kernel family registry, external-buffer ABI roles, and dispatch-policy defaults shared by Metal MSL source planning, OpenCL source artifacts, Vulkan SPIR-V binding, and MPSRT runtime-model generation
- `src/kernel_ir/gfx_opencl_source_artifacts.*`: baseline OpenCL source kernels and their role-based manifest contracts
- `src/mlir/`: MLIR support probes, Apple stage-pipeline passes, Apple vendor descriptor contracts, typed MPSRT dialect/materialization helpers, MPSRT builder-plan materialization, backend custom-kernel ABI adapters, split Apple MSL/MPS source-plan helpers, MatMul Metal/MPSRT helpers, SPIR-V binding adapters, runtime-value helpers, runtime-parameter payload helpers, const-tensor-source attachment, and shared codegen helpers
- `src/backends/metal/`: Metal-specific plugin glue, runtime, memory, profiling, MSL compilation, and MPSGraph-backed vendor stages
- `src/backends/opencl/`: OpenCL-specific plugin glue, dynamic runtime loader, buffer manager, program cache, source-stage execution, and stubs
- `src/backends/vulkan/`: legacy Vulkan-specific plugin glue, runtime, buffers, profiling, SPIR-V/Vulkan execution
- `src/transforms/`: OpenVINO graph passes and fusion logic
- `src/kernel_ir/`: shared kernel metadata and planning structures
- `tests/`: unit, integration, backend, and tooling coverage
- `tools/`: developer scripts for profiling workflows, microbench smoke checks, and report post-processing
- `docs/`: local module docs, including profiling and microbench references
- `third_party/llvm-project/`: vendored LLVM/MLIR used by the build
- `third_party/Vulkan-Headers/`: Vulkan-Headers git submodule used by the Raspberry Pi Vulkan toolchain flow

## Main Runtime Components
### Plugin
`src/plugin/plugin.cpp` owns:
- property parsing and validation
- backend resolution through `GFX_BACKEND`
- `query_model()`
- remote context creation
- model transformation before compilation
- backend-aware transform pipeline selection, including backend-specific protection of decompression subgraphs when a later runtime route depends on them
- backend-specific pattern fusion such as LLaMA rotate-half rewriting into native `RoPE` on Metal and horizontal regrouping of compatible compressed `MatMul` nodes

### CompiledModel
`src/plugin/compiled_model.cpp` owns:
- compiled-model properties and profiling configuration
- backend state creation
- stage pipeline construction in `build_op_pipeline()`
- optional fusion of compatible stages through `FusedSequenceStage`
- absorption of selected transpose inputs into downstream stages through `GfxInputTransform`
- output-source tracking for fused or rewritten stages so public outputs and direct stateful-assign consumers stay mapped to the original OpenVINO node/port
- output-alias tracking when one runtime stage materializes several graph outputs through the same underlying buffer
- per-stage optimization planning that now includes placement domain and storage decisions, not only fusion and submit-policy hints

### InferRequest
`include/openvino/gfx_plugin/infer_request.hpp` plus backend-specific implementation files own:
- host and remote tensor binding
- per-request backend state
- command submission
- profiling collection
- per-request stateful variable storage for `ReadValue` / `Assign`

The current infer path is not a naive "execute one stage, submit immediately" loop. `src/plugin/infer_submission.*` and `src/plugin/infer_pipeline.*` now provide:
- reusable bound pipelines and prepared input-resolution plans
- prepared output-resolution plans for stage outputs, passthrough parameters, and materialized constant outputs
- reusable host output tensors for static output signatures when the user does not bind explicit output storage
- submission windows driven by stage submit policy, stage count, and output-byte thresholds
- direct dependency-aware window extension for producer-consumer chains that would otherwise be split only because a soft stage/output/MAC budget was reached
- backend-specific submission sessions for Metal, OpenCL, and Vulkan
- self-healing reusable host outputs that are recreated if a cached host tensor no longer matches the required type or shape
- workspace-managed stage-output allocation that can recycle intermediate buffers across compatible stage lifetimes and report slot usage through profiling counters
- stateful `Assign` prebinding that can bind persistent variable outputs through source-node-aware pipeline output links instead of forcing a later copy-only path

### Backend-neutral runtime
`src/runtime/` contains:
- `GpuStage`: execution-stage interface
- `GpuStageFactory` / `ExecutionDispatcher`: backend-specific stage dispatch
- `gfx_stage_policy.*`: runtime route, fusion, and submit-policy selection
- `gfx_stage_policy.*` now also selects placement domains such as `apple_mps`, `apple_msl`, `opencl`, and `spirv`, together with storage kinds such as `image`, `matrix`, and `buffer`
- manifest-backed stage metadata now distinguishes vendor-primitive stages from custom-kernel stages and carries stable kernel-family, semantic input/output roles, dispatch policy, and external-buffer-ABI contracts across the MLIR, compile, and runtime layers
- `gfx_mpsrt_model.*` now lives in `src/runtime/` and owns runtime-model reconstruction, resource-table finalization, tensor binding plans, and external-buffer ABI adaptation shared by Metal preparation/tests rather than a Metal-only model object
- generated `gfx_mpsrt_ops` metadata and typed builder-plan records now serialize multi-stage model records, external-buffer roles, per-stage ABI descriptors, runtime resources, and storage bridges into the Metal compile path instead of relying on flat legacy `gfx.mpsrt.*` attrs for stage reconstruction
- storage bridges now cover not only image-backed stages but also matrix, ndarray, and alias-style contracts where the typed program/runtime model needs an explicit conversion edge
- `gfx_parallelism.*` and `gfx_partitioning.*`: backend-neutral device-capability and workgroup planning helpers
- `gfx_target_profile.*`: shared profiling snapshot for the resolved backend, device family, workgroup limits, memory-alignment traits, and kernel capability flags
- `immutable_gpu_buffer_cache.*`: backend-neutral cache for immutable device buffers
- shared remote context/tensor abstractions
- common tensor, buffer, logging, kernel-binding, and parallelism helpers

`GpuStage` now exposes two hooks that affect real runtime behavior:
- `set_input_transform()`: lets a stage consume an absorbed transpose as metadata instead of materializing a separate runtime stage
- `submit_policy()`: lets a stage communicate scheduling weight or isolation requirements to the infer pipeline

The runtime also has explicit reuse layers:
- immutable constant payloads can be cached as device buffers through backend const-cache implementations
- compiled kernels can reuse prepared binding tables through shared backend-neutral cache helpers in `gpu_backend_base.hpp`
- prepared binding-table caches can grow past their initial size when the infer path observes more distinct reusable binding sets
- infer requests can reuse prepared output bindings and preallocated host output tensors across repeated executions
- on Metal, MSL-dispatch stages can also be wrapped as compact MPSRT runtime models with explicit external-buffer ABI roles, prepared pipeline-cache entries, typed builder-plan records, and request-time binding through a resource table
- on Metal, MPSRT execution can now cover vendor-only plans such as `MPSGemm`, `MPSCNNConvolution`, `MPSCNNPooling*`, `MPSImageBilinearScale`, `MPSMatrixSoftMax`, `MPSMatrixFindTopK`, and MPSGraph-backed GEMM / TopK / SDPA cases, plus hybrid multi-stage plans such as `MPSGemm + MSL epilogue`
- on Metal, prepared MPSRT models now classify resources as external, model-owned, or transient; single-stage MSL dispatch keeps const/runtime-parameter ABI buffers external, while vendor/typed owned constants use the prepared-resource table
- on Metal, transient buffer/image resources are allocated from a prepared Metal heap with live-window aliasing

Profiling now also has two layers:
- compile-time tracing stored as a JSON `compile` section inside `GFX_PROFILING_REPORT`
- infer-time node, segment, transfer, allocation, and counter reporting through `gfx_profiling_report.*`
- infer `stage_execute` segments now also carry lightweight data-movement and compute estimates used by the extended roofline-style summaries
- the active target profile is stored in `extended.target_profile` and mirrored through counters such as `target_backend_opencl`, `target_backend_metal`, and `target_backend_vulkan`

Backend-neutral planning now consumes device info exported by the active buffer manager:
- Metal, OpenCL, and Vulkan buffer managers report subgroup width, workgroup limits, and device family through `GpuExecutionDeviceInfo`
- `gfx_parallelism.*` converts that into execution-policy caps
- `gfx_partitioning.*` derives 1D and 2D workgroup shapes from the same data

Current family-aware planning distinguishes at least:
- `apple` for Metal
- `adreno` for Qualcomm Vulkan devices
- `broadcom_v3d` for Raspberry Pi Vulkan devices
- `generic` as the fallback class

Family-aware tuning is now used for more than cache keys. The current code includes:
- Broadcom V3D-specific matmul and convolution parallelism choices for Raspberry Pi-style Vulkan devices, including occupancy-aware 64-thread pointwise/light Conv groups and 128-thread caps for ultra-dense Conv groups
- capability-gated Conv2D output-channel blocking and optional spatial micro-tiling for backends that report support through `GpuExecutionDeviceInfo`
- plain Vulkan Conv2D and GroupConv use the shared canonical MLIR builders plus common convolution lowering and no longer route through executor-level direct/chunked convolution kernels
- MLIR convolution lowering that can honor explicit dispatch tile and thread attributes emitted by the planning path
- MLIR convolution lowering that now uses a faster full-tile path for interior tiles and falls back to lane guards only on edge tiles

## Backend Selection
The plugin has two layers of backend choice:
- Build-time availability via CMake:
  - `GFX_ENABLE_METAL`
  - `GFX_ENABLE_OPENCL`
  - `GFX_ENABLE_VULKAN`
  - `GFX_DEFAULT_BACKEND`
- Runtime selection via property:
  - `GFX_BACKEND=metal`
  - `GFX_BACKEND=opencl`
  - `GFX_BACKEND=vulkan`

On macOS, CMake disables OpenCL and Vulkan for this module and Metal becomes the only runtime backend. On non-Apple builds, `auto` resolves to OpenCL first when the OpenCL source backend is enabled; Vulkan is selected only when OpenCL is unavailable or explicitly requested for diagnostics.

## Supported Ops
Support is driven by MLIR builders in `src/mlir/` and backend runtime implementations. The active set includes:
- MatMul, Conv2D, Conv3D, GroupConv
- Add, Sub, Mul, Div, Pow, Mod, FloorMod
- compare, logical, and select operations
- unary activations and elementwise transforms
- RMSNorm-style lowered `RMS`
- rotary positional embeddings through native `RoPE`
- backend-specific `ScaledDotProductAttention` support on Metal for rank-4 FP16/FP32 Q/K/V inputs
- backend-specific fused causal-mask `ScaledDotProductAttention` support on Metal for selected LLM graphs with explicit `attention_mask` and `cache_position` inputs
- reduction ops such as ReduceSum, ReduceMean, ReduceMax, ReduceMin, ReduceProd, ReduceL1, and ReduceL2
- MaxPool, AvgPool, Softmax, BatchNormInference
- Concat, Split, Slice, Transpose, Reshape, Convert, Interpolate
- Gather, GatherND, GatherElements, ScatterUpdate, ScatterElementsUpdate, ScatterNDUpdate, ShapeOf, Range, Tile, TopK, SpaceToDepth, DepthToSpace
- stateful graph ops through `ReadValue` / `Assign`

Important constraints:
- many paths require static rank or static shape
- many ops require constant attributes
- backend parity is not guaranteed between Metal, OpenCL, and Vulkan

Current lowering/runtime special cases:
- selected transpose inputs can be absorbed into Add, Conv2D, GroupConv2D, and Split lowering instead of staying as standalone runtime stages
- transformation now depends on the resolved backend, so compile/query can preserve or decompose backend-sensitive patterns differently
- OpenCL currently executes baseline source artifacts for f32 linear copy/layout, dynamic f16 data movement, f32/i32/i64 convert casts, MatMul, Softmax, transpose, slice, strided-slice, Range, Tile, gather, gather-elements, gather-nd, scatter-update, scatter-elements, scatter-nd, ShapeOf, Concat, Split, VariadicSplit, unary elementwise, same-shape/scalar/broadcast binary elementwise, same-shape/broadcast compare, same-shape/broadcast Select, boolean logical, and boolean logical-reduction families when their artifact contracts match
- OpenCL source artifacts are intentionally small and role-based: constant tensor operands can be materialized by the source stage, boolean buffers are padded for aligned 32-bit stores, dynamic runtime dimensions are passed through scalar ABI metadata, and portable kernels avoid pointer-selection and wide-integer assumptions that are not guaranteed on mobile-class OpenCL stacks
- Vulkan contains specialized direct or chunked paths for unary, binary, softmax, split/concat, transpose, and convert cases; Conv2D and GroupConv2D are intentionally kept on the shared MLIR/SPIR-V custom-kernel path
- Vulkan now also has specialized chunked paths for `RMS` and binary `Concat`
- decomposed Conv2D routes such as `im2col + matmul` are not an active production route; convolution policy must stay on a single typed custom-kernel or a future typed multi-kernel manifest
- Softmax lowering now supports arbitrary normalized axes instead of only the last axis
- Reduce lowering now extracts axes and `keep_dims` through concrete Reduce op types instead of relying on a generic reduction base path
- transform cleanup now runs OpenVINO `RMSFusion` before plugin-local cleanup, so common RMSNorm tails can reach dedicated `RMS` lowering and backend codegen
- Metal `RMS` stages can also fuse one residual `Add` input directly into the RMS kernel when the shape contract matches
- Metal can keep compressed `MatMul` weight decompression subgraphs intact so later stage compilation can use backend-side compressed or repacked constant paths instead of losing that structure during generic transforms
- Metal can also fuse the common LLaMA rotate-half arithmetic pattern into a native `RoPE` stage when the resulting RoPE layout constraints are supported
- Metal can fuse selected LLM causal-mask attention graphs into a native `ScaledDotProductAttention` variant that consumes `attention_mask` and `cache_position`; Vulkan still rejects that native path today
- compatible compressed `MatMul` nodes that share the same data input can be regrouped into one horizontally fused `MatMul` followed by `VariadicSplit`
- compressed `MatMul` stage compilation can repack concatenated quantized weight parts and scale blocks into backend const buffers before codegen
- Metal custom MSL source generation and runtime binding metadata are centralized in MLIR source-plan helpers, including compressed `MatMul`, SDPA, and scalar/runtime-parameter-heavy custom kernels
- Apple MSL source planning is split into focused `msl_codegen_apple_msl_*` files for binding, dispatch, elementwise, convolution, matmul, pool, reduction, shape, slice, concat/split, TopK, unary, and LLM-style helpers, while Apple MPS, attention, compressed MatMul, and MatMul MPSRT routes live in their dedicated source-plan files
- Metal request-time binding follows `GfxMslRuntimeBindingPlan` / manifest external-buffer roles, so tensor inputs, outputs, scalar params, runtime params, and const tensors keep an explicit kernel-buffer order instead of relying on positional conventions
- Vulkan/MLIR compact ABI paths use `spirv_kernel_binding_adapter.hpp` to keep fixed SPIR-V argument metadata separate from Apple MSL manifest binding metadata
- Metal Conv2D and MaxPool codegen now honor dilation metadata, and Conv2D dispatch planning can block output channels and output width per thread for selected float-like cases
- `ShapeOf`, `TopK`, `Tile`, and unary stage handling now have stricter runtime/codegen paths around output typing, alias safety, and ABI metadata
- Metal stage policy now routes selected 4D conv/pool/interpolate-style stages to Apple MPS image storage, selected `MatMul` / last-dimension `Softmax` / `TopK` stages to Apple MPS matrix storage, and keeps the remaining cases on Apple MSL buffer dispatch
- the Metal MSL path now carries a custom-kernel family manifest plus explicit external-buffer ABI roles and dispatch policy, so runtime parameter, input, and output buffers can be rebound without assuming a simple tail-output convention
- Metal MPSRT stages now materialize explicit storage bridges such as `buffer_to_image`, `image_to_buffer`, `buffer_to_matrix`, `matrix_to_buffer`, `buffer_to_ndarray`, and `alias` when public bindings cross a typed storage boundary
- Metal MatMul lowering can now choose a vendor MPS GEMM route directly, or a mixed `MPSGemm + MSL epilogue` route when bias or supported activation fusion still needs a custom kernel stage
- Metal source planning can now pick `SingleStage` or `MultiStage` MPSRT kernel-source plans and attach constant tensor payloads for vendor convolution-family stages before request-time execution
- Metal MPSRT metadata materialization now prefers a typed `GfxMpsrtProgram` plus generated `gfx_mpsrt_ops` function and explicitly cleans stale legacy `gfx.mpsrt.*` stage attrs after materialization
- Apple MPS stage construction now runs through a dedicated Apple stage pipeline and shared vendor descriptor helpers instead of ad-hoc per-op metadata assembly
- selected static NCHW bilinear `Interpolate` stages can route to Apple MPS `Resize2D` image storage through the same MPSRT descriptor, storage-bridge, and resource-binding path as other vendor primitives
- selected rank-4 FP16/FP32 attention patterns can route to an MPSGraph SDPA vendor stage through the same MPSRT resource table and external-buffer binding contract
- MPSRT request binding now follows an explicit resource table and external-buffer binding list, so MSL dispatch external const/runtime-parameter buffers, model-owned vendor constants, transient tensors, and image bridge resources are validated before encoding
- Metal Conv2D manifest handling now also covers the legacy custom `conv2d_kernel` family under the same manifest/ABI contract used by other MSL dispatch kernels
- `ScatterUpdate` now has a dedicated MLIR lowering path instead of falling back to the older scatter-family builders
- Slice lowering now prefers `tensor.extract_slice`; generic slice metadata extraction still accepts the older generic form when needed
- dynamic-shape support now covers `ShapeOf` compile/query flow and query-time acceptance for selected data-movement ops such as `Concat`, `Broadcast`, `Select`, `StridedSlice`, and `Range`
- `ReadValue` is treated as a view-style stage, while `Assign` is intercepted by a stateful execution layer that persists the variable buffer inside infer-request state
- Metal dynamic-shape `MatMul` can repack a constant RHS from `f32` to `f16` during stage compilation and then compile the kernel against the effective runtime buffer types instead of only the original node input element types
- Multiply-style eltwise stages can now fuse selected activations into one chosen input instead of only fusing an activation on the final stage output
- contiguous `Split` and `VariadicSplit` outputs can alias byte ranges of the input buffer instead of forcing materialized copies when the split layout is view-compatible
- layout cleanup can fold the DFL softmax expectation tail into a value-preserving `Softmax -> MatMul -> Reshape/Transpose` path instead of the older synthetic convolution rewrite

## Public And Internal Properties
Commonly used properties:
- `GFX_BACKEND`
- `GFX_ENABLE_FUSION`
- `GFX_PROFILING_LEVEL`
- `GFX_PROFILING_REPORT`
- `GFX_MEM_STATS`
- `GFX_DIAGNOSTIC_F32_MPS_IMAGE`
- `ov::hint::inference_precision`
- `ov::available_devices`
- `ov::device::id`
- `ov::cache_dir`
- `ov::enable_profiling`
- `ov::loaded_from_cache`
- legacy `PERF_COUNT`

See `src/plugin/gfx_property_lists.cpp` for the exact supported property sets exposed by the plugin and by compiled models.

Practical meanings:
- `GFX_BACKEND`: request `metal`, `opencl`, or `vulkan`
- `GFX_ENABLE_FUSION`: enable stage fusion during pipeline construction
- `GFX_PROFILING_LEVEL`: control profiling detail level
- `GFX_PROFILING_REPORT`: fetch the latest profiling report, including compile and infer sections when profiling is enabled
- `GFX_MEM_STATS`: fetch backend memory statistics from a compiled model
- `GFX_DIAGNOSTIC_F32_MPS_IMAGE`: diagnostic Metal compile switch for localizing selected `f32` Conv/GroupConv/Pool image placement through the normal stage planner
- `ov::hint::inference_precision`: select `f16` or `f32` GFX inference precision; the default is `f16`
- `ov::available_devices`: expose stable numeric device ids such as `"0"`; use `ov::device::full_name` for the human-readable backend device name
- `ov::device::id`: select one of the numeric device ids exposed through `ov::available_devices`
- `ov::cache_dir`: reuse the standard OpenVINO cache directory for Vulkan pipeline-cache persistence
- `ov::loaded_from_cache`: report whether the OpenVINO model-cache path loaded this compiled model

## Build
Assuming an OpenVINO Developer Package is available:

```bash
cd /path/to/gfx_plugin
cmake -S . -B build-gfx-plugin -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DOpenVINODeveloperPackage_DIR=/path/to/openvino/install/cmake \
  -DENABLE_TESTS=ON \
  -DGFX_DEFAULT_BACKEND=auto
cmake --build build-gfx-plugin --target openvino_gfx_plugin ov_gfx_func_tests
cmake --build build-gfx-plugin --target ov_gfx_unit_tests ov_gfx_runtime_micro_tests ov_gfx_microbench
```

Useful options:
- `-DGFX_ENABLE_METAL=ON|OFF`
- `-DGFX_ENABLE_OPENCL=ON|OFF`
- `-DGFX_ENABLE_VULKAN=ON|OFF`
- `-DGFX_DEFAULT_BACKEND=auto|metal|opencl|vulkan`

Build notes:
- vendored LLVM/MLIR is now built as a static external toolchain under `third_party/llvm-project`
- the external LLVM bootstrap now injects a tiny local dummy fuzzing-engine archive so `mlir-parser-fuzzer` configure paths do not break the bundled llvmorg-22.1.2 flow
- Android and generic cross-compiling flows forward toolchain settings into that external LLVM/MLIR build
- that external LLVM bootstrap also forwards `CMAKE_C_FLAGS`, `CMAKE_CXX_FLAGS`, and the executable/shared/module linker flag families into the nested LLVM configure step
- when cross-compiling, the nested LLVM native-tool bootstrap also receives host compiler/sysroot flags through `CROSS_TOOLCHAIN_FLAGS_NATIVE` / `CROSS_TOOLCHAIN_FLAGS_LLVM_NATIVE`; LLVM tools are disabled because the plugin only needs the MLIR libraries for this external build
- the module build treats compiler warnings as errors by default through `-Werror` on Clang/GCC and `/WX` on MSVC
- `cmake/GfxAndroidRuntimeBundle.cmake.in` provides helper copy logic for Android-side runtime dependency bundling
- `third_party/Vulkan-Headers` is tracked as a git submodule pinned to the module-tested upstream release
- Metal now builds the local MPSRT execution sources under `src/backends/metal/runtime/mpsrt/` together with the shared `src/runtime/gfx_mpsrt_model.*`, `gfx_mpsrt_*`, `gfx_kernel_manifest.hpp`, and `gfx_custom_kernel_families.*` helpers; Apple MSL source planning is compiled into the shared MLIR runtime library rather than the Metal runtime-only target
- Metal backend detection now also requires `MetalPerformanceShaders.framework` and `MetalPerformanceShadersGraph.framework`, not only `Metal.framework` and `Foundation.framework`
- OpenCL backend sources build without linking a compile-time OpenCL SDK; the runtime dynamically loads the target OpenCL library and reports the selected GPU through `GpuExecutionDeviceInfo`
- `tools/gfx_rpi_vulkan_toolchain_builder.py` can assemble a hermetic Raspberry Pi Vulkan cross-toolchain bundle for Raspberry Pi 4/5 class `aarch64` Bookworm-style targets, normalize absolute sysroot symlinks, install both `vulkan/` and `vk_video/` headers into the generated sysroot, and emit portable `-march=armv8-a` compile flags in the generated wrappers and toolchain file

The build produces the `openvino_gfx_plugin` shared library. On Unix-like builds this is typically emitted as `libopenvino_gfx_plugin.so`; the `.so` suffix is also forced on macOS for OpenVINO plugin loading compatibility.

## Quick Start
For a local development build, register the plugin explicitly through `ov::Core`.

### Register the plugin
```cpp
#include <openvino/openvino.hpp>

ov::Core core;
core.register_plugin("/path/to/libopenvino_gfx_plugin.so", "GFX");
```

### Inspect the active backend
```cpp
std::string backend = core.get_property("GFX", "GFX_BACKEND").as<std::string>();
std::string full_name = core.get_property("GFX", ov::device::full_name).as<std::string>();
```

### Compile a model for GFX
```cpp
auto model = core.read_model("/path/to/model.xml");

ov::AnyMap config = {
    {"GFX_BACKEND", "metal"},
    {"GFX_ENABLE_FUSION", true},
    {ov::enable_profiling.name(), true},
};

ov::CompiledModel compiled = core.compile_model(model, "GFX", config);
```

### Run inference
```cpp
ov::InferRequest request = compiled.create_infer_request();
request.set_input_tensor(input_tensor);
request.infer();
ov::Tensor output = request.get_output_tensor();
```

For longer examples and property-oriented workflows, see `docs/USAGE.md`.

## Running Tests
Run the labeled suite:

```bash
ctest --test-dir build-gfx-plugin --output-on-failure -L GFX
```

Locate and run the gtest binaries directly:

```bash
find build-gfx-plugin -name ov_gfx_func_tests -type f
DYLD_LIBRARY_PATH=/path/to/openvino/runtime/libs \
  <path-to-ov_gfx_func_tests> --gtest_filter=MetalBasicOps.*

find build-gfx-plugin -name ov_gfx_unit_tests -type f
DYLD_LIBRARY_PATH=/path/to/openvino/runtime/libs \
  <path-to-ov_gfx_unit_tests> --gtest_filter=GfxMlirTransforms.*
```

`ov_gfx_func_tests` covers plugin-facing and functional behavior, `ov_gfx_unit_tests` carries focused runtime, MLIR, backend, and cache regressions, and `ov_gfx_runtime_micro_tests` is used for smaller runtime subgraph checks. Helper tools `ov_gfx_compare_runner` and `ov_gfx_microbench` are also built from `tests/tools/`.

Recent regression coverage includes:
- Metal stage-placement and MPSRT descriptor/runtime-model coverage in `tests/unit/gfx_stage_policy_test.cpp`
- Metal backend tests for MPSRT compile, prepared-pipeline caching, and request-time MSL dispatch execution in `tests/backends/metal/gpu_backend_test.mm`
- Metal backend tests now also cover manifest-driven buffer ordering, program-to-ops materialization, runtime-parameter roles, typed builder-plan/runtime-model execution, storage bridges, MPSRT resource-table binding, prepared resource heaps, vendor `MPSGemm` / Conv2D / Pool2D / Resize2D / Softmax / TopK / SDPA execution, MPSGraph executable paths, and hybrid prepared-model execution
- Metal MSL binding-plan tests cover compressed `MatMul`, SDPA, scalar/runtime-parameter ordering, manifest-derived argument counts, output-before-runtime-params ABI ordering, and request-time rejection when an MSL dispatch stage is missing materialized kernel-buffer order
- OpenCL source-artifact tests in `tests/unit/gfx_opencl_source_artifacts_test.cpp` cover the baseline source manifest and role ABI contracts
- SPIR-V binding-adapter tests cover compact fixed-argument metadata without reusing legacy Apple MSL operand/scalar attrs
- canonical Conv2D MLIR lowering checks
- strict interior-tile bounds checks plus Vulkan batch-1 parallel-launch, batch>1 serial-fallback, and Pool2D parallel-dispatch checks in `tests/unit/mlir_conv_parallel_test.cpp`
- regression coverage that keeps decomposed Conv2D experiments out of the production lowering path until a typed multi-kernel manifest exists
- linear matmul parallel-lowering coverage
- ReduceSum builder coverage in `tests/unit/basic_ops_internal_test.cpp`
- absorbed input-transform tests for Add, Conv2D, GroupConv2D, and Split
- Vulkan runtime regression coverage in `tests/backends/vulkan/`
- infer submission, prepared-pipeline reuse, immutable-const-cache reuse, and shared kernel-binding reuse tests
- dependency-aware submit-window extension coverage in `tests/unit/infer_submission_test.cpp`, including direct producer-consumer chains and boundaries that must still force a new window
- Vulkan batched constant-upload behavior through the shared infer command buffer path
- Broadcom-oriented dense stride-1, huge-spatial, ultra-dense, and output-channel-blocked convolution tuning coverage in `tests/unit/gfx_parallelism_test.cpp`
- explicit shared-test plugin registration coverage: the functional test target removes implicit `plugins.xml`, registers GFX/TEMPLATE directly where needed, and keeps uninstantiated shared-test allow-listing centralized in `tests/gfx_shared_gtest_allow.cpp`
- reusable output-resolution and reusable host-output coverage in `tests/unit/infer_pipeline_reuse_test.cpp`
- internal transform and plugin coverage in `tests/unit/basic_ops_internal_test.cpp`
- DFL softmax expectation rewrite coverage, including value-preservation checks for the MatMul form, in `tests/unit/layout_cleanup_test.cpp`
- backend memory/device integration coverage in `tests/unit/memory_device_integration_test.mm`
- plugin property coverage for numeric `available_devices` / `ov::device::id` behavior in `tests/unit/plugin_tests.cpp`
- target-profile profiling coverage in `tests/unit/gfx_profiling_report_test.cpp`
- precision-aware accuracy tolerance helpers in `tests/gfx_accuracy_tolerance.hpp`, shared by the compare runner and shared-test instances
- `ov_gfx_conv_shape_bench` in `tests/tools/` for compile-plus-infer sweeps across representative YOLO26x convolution shapes on `GFX`; `CPU` is only a separate performance orienter
- standalone OpenCL Conv2D microbench tools in `tests/tools/` for deciding whether an OpenCL kernel family is worth promoting into the shared GFX manifest/runtime contracts

## Debugging And Instrumentation
Useful environment variables from the current codebase:
- `OV_GFX_TRACE`: trace logging
- `OV_GFX_DEBUG`: debug logging
- `OV_GFX_TEST_DEBUG`: extra test-side logging
- `OV_GFX_DEBUG_MSL`: dump generated Metal shader sources
- `OV_GFX_SAFE_DEBUG`: enable additional Metal memory safety checks
- `OV_GFX_DUMP_SPIRV_BINDINGS`: dump Vulkan binding information
- `OV_GFX_DUMP_SPIRV_MLIR`, `OV_GFX_DUMP_SPIRV_MLIR_FILTER`, `OV_GFX_DUMP_MLIR_PRE_SPIRV`: Vulkan/MLIR dump controls

For output-quality checks against a reference backend, use `ov_gfx_compare_runner`. It is an accuracy-only helper: it registers local plugin builds, compares tensor diffs, can run per-op windows or full-graph per-op output scans, can feed reproducible RGB PPM images through `--input-image`, can compare against precomputed `--golden-dir` outputs, and can also emit `GFX`-only output summaries for quick debugging. Diff reports now identify outputs by `friendly_name:port` and print the first threshold violation, the worst absolute mismatch, and reference/GFX values. `--per-op-all` is a real gate: it reports `PER_OP_FIRST_MISMATCH` / `PER_OP_MISMATCH` and exits non-zero when any observed internal output exceeds the precision-aware threshold policy, while `PER_OP_MATCH max_abs=... max_rel=... tolerance_violations=0` records a clean scan. Use `--gfx-inference-precision f16|f32` to exercise the GFX compile property during diagnostics. For performance numbers, use `benchmark_app` instead of the compare tool.

Per-op and debug-output modes (`--per-op`, `--per-op-all`, `--single-op-output`) compile GFX with `GFX_ENABLE_FUSION=false` through the normal compile-property mechanism so diagnostic outputs map to original OpenVINO op boundaries. Regular full-graph compilation keeps the configured fusion policy.

For profiling-driven triage, use `ov_gfx_microbench` plus the local references in `docs/MICROBENCH_SCHEMA.md` and `docs/PROFILING_RUNBOOK.md`.

## Integration Notes
- `query_model()` follows the same backend-specific support checks as compilation, so external schedulers see actual backend capability rather than an optimistic superset.
- Unsupported models fail during compilation instead of silently mixing CPU execution into a request.
- `import_model()` re-creates an OpenVINO model and recompiles it for the resolved backend.
- `export_model()` writes model data, not a serialized GPU pipeline cache.
- Remote contexts and remote tensors are available, but practical capabilities still depend on the active backend implementation.
- Runtime submissions and constant-buffer reuse are backend-aware internals; they improve execution behavior but do not change the plugin's external OpenVINO contract.
- Output binding now has a reusable internal planning layer for stage outputs, passthrough outputs, and constant outputs, but the public infer-request API remains standard OpenVINO.

## Developer Documentation
All developer-facing documentation intended for publication from this directory lives under `docs/`:
- `docs/ARCHITECTURE.md`
- `docs/DEVELOPMENT.md`
- `docs/TESTING.md`
- `docs/USAGE.md`

If you add new documentation meant for published consumers of this module, keep it in English and keep it inside `modules/gfx_plugin/`.

## Current Limitations
- The plugin is still experimental.
- Dynamic-shape support is partial.
- There is no partial CPU fallback path.
- Remote tensor support exists but is not yet a fully polished cross-platform feature set.
- Backend coverage and runtime maturity differ between Metal, OpenCL, and Vulkan.
- Some optimized OpenCL and Vulkan paths depend on device capabilities such as subgroup size, memory alignment, and compute workgroup limits.

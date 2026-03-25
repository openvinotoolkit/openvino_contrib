# Development Guide

This document is for contributors working inside `modules/gfx_plugin`.

## Prerequisites
- CMake 3.13 or newer
- an OpenVINO Developer Package build
- Ninja recommended
- Metal toolchain on macOS for the Metal backend
- Vulkan SDK or system Vulkan development files for the Vulkan backend

The module vendors LLVM/MLIR under `third_party/llvm-project` and can build the required MLIR pieces as part of the CMake flow.

## Configure And Build
Example configuration:

```bash
cd /path/to/gfx_plugin
cmake -S . -B build-gfx-plugin -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DOpenVINODeveloperPackage_DIR=/path/to/openvino/install/cmake \
  -DENABLE_TESTS=ON \
  -DGFX_DEFAULT_BACKEND=auto
```

Build:

```bash
cmake --build build-gfx-plugin --target openvino_gfx_plugin ov_gfx_func_tests
```

Useful CMake options:
- `GFX_ENABLE_METAL`
- `GFX_ENABLE_VULKAN`
- `GFX_DEFAULT_BACKEND`
- `ENABLE_TESTS`

On macOS, Vulkan is disabled by `cmake/GfxBackendConfig.cmake`.

## Where To Start Reading
Read in this order:
1. `README.md`
2. `docs/ARCHITECTURE.md`
3. `src/plugin/plugin.cpp`
4. `src/plugin/compiled_model.cpp`
5. `include/openvino/gfx_plugin/infer_request.hpp`
6. the backend directory you are changing

For runtime execution, follow:
- `Plugin::compile_model()`
- `CompiledModel::build_op_pipeline()`
- backend-specific `infer_*_impl()`
- backend stage factory and executor code

If the behavior depends on route or scheduling selection, also read:
- `src/runtime/gfx_stage_policy.*`
- `src/runtime/gfx_parallelism.*`
- the active backend executor, especially under `src/backends/vulkan/runtime/`

If the change touches infer-request throughput or resource reuse, also read:
- `src/plugin/infer_submission.*`
- `src/plugin/infer_pipeline.*`
- `src/runtime/immutable_gpu_buffer_cache.*`
- `src/runtime/gpu_backend_base.hpp`
- `src/plugin/infer_io_utils.*`

## Adding Or Extending An Op
Typical path:
1. Add or extend MLIR support in `src/mlir/`
2. Ensure support probing succeeds through `mlir_supports_node()`
3. Implement or update backend runtime/codegen handling
4. Make sure the relevant backend stage can be created
5. Add tests in `tests/unit/` and the relevant backend test directory

For the current codebase, also check whether the op should participate in:
- absorbed input transforms through `GfxInputTransform`
- stage policy selection in `src/runtime/gfx_stage_policy.*`
- MLIR parallel lowering or cleanup passes
- backend-specialized fast paths such as Vulkan direct or chunked routes
- reusable bytes-arg materialization or immutable const-buffer caching

If the op needs fusion support, inspect:
- `src/transforms/`
- `src/runtime/fused_sequence_stage.*`
- fusion planning inside `src/plugin/compiled_model.cpp`

## Properties
Supported property lists are defined in:
- `src/plugin/gfx_property_lists.cpp`
- `include/openvino/gfx_plugin/properties.hpp`

If you add a property:
1. define the property key if needed
2. add it to the supported property list
3. parse and validate it in plugin or compiled-model code
4. cover it with tests

## Debugging
Useful environment variables:
- `OV_GFX_TRACE`
- `OV_GFX_DEBUG`
- `OV_GFX_TEST_DEBUG`
- `OV_GFX_DEBUG_MSL`
- `OV_GFX_SAFE_DEBUG`
- `OV_GFX_DUMP_SPIRV_BINDINGS`
- `OV_GFX_DUMP_SPIRV_MLIR`
- `OV_GFX_DUMP_SPIRV_MLIR_FILTER`
- `OV_GFX_DUMP_MLIR_PRE_SPIRV`

These are implemented directly in the runtime and codegen sources; grep for `OV_GFX_` when adding new diagnostics.

For functional comparison against a reference backend, build and run `tests/tools/ov_gfx_compare_runner.cpp`. It is useful when a new path changes numerics or when you need a per-op comparison window instead of only a gtest suite.

For reuse and submission changes, prefer the focused unit tests under:
- `tests/unit/infer_submission_test.cpp`
- `tests/unit/infer_pipeline_reuse_test.cpp`
- `tests/unit/gpu_const_cache_test.cpp`
- `tests/unit/kernel_arg_reuse_test.cpp`
- `tests/unit/gpu_backend_base_test.cpp`

`tests/unit/infer_pipeline_reuse_test.cpp` now also covers:
- prepared output-resolution plans
- reusable host output tensors for static outputs
- reuse of pre-resolved input vectors across repeated executes

## Documentation Rules For This Module
- Keep published module docs in English
- Keep module docs inside `modules/gfx_plugin/`
- Update `README.md` when user-visible behavior changes
- Update `docs/ARCHITECTURE.md` when the code-truth architecture changes
- Update `docs/TESTING.md` when test layout or commands change

## What Not To Do
- do not rely on silent CPU fallback
- do not document removed architectures as current behavior
- do not put primary module documentation only in repository-level files outside this directory
- do not leave backend-specific route changes undocumented when they affect supported shapes, layout assumptions, or profiling behavior
- do not add ad-hoc backend caches before checking whether `ImmutableGpuBufferCache` or the shared prepared-binding cache already solves the problem
- do not bypass reusable output planning when changing infer I/O paths; keep stage outputs, passthrough outputs, and constant outputs on the same documented resolution path

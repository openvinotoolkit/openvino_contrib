# GFX Plugin Architecture

This document describes the current architecture implemented in `modules/gfx_plugin`.

## Design Goals
- expose a single OpenVINO device named `"GFX"`
- share as much logic as possible between Metal and Vulkan
- keep backend-specific code isolated under `src/backends/`
- reject unsupported models early instead of silently falling back to CPU

## Top-Level Structure
- `src/plugin/`: OpenVINO-facing integration
- `src/runtime/`: backend-neutral runtime interfaces and helpers
- `src/mlir/`: MLIR support probing and lowering helpers
- `src/backends/metal/`: Metal-specific plugin, runtime, memory, profiling, and codegen pieces
- `src/backends/vulkan/`: Vulkan-specific plugin, runtime, profiling, and codegen pieces
- `src/transforms/`: graph rewrites and fusion logic
- `src/kernel_ir/`: shared kernel metadata and planning structures

## Main Objects
### `ov::gfx_plugin::Plugin`
Defined through `src/plugin/plugin.cpp` and the public header in `include/openvino/gfx_plugin/plugin.hpp`.

Responsibilities:
- register the `GFX` device
- validate and normalize properties
- resolve the backend (`metal` or `vulkan`)
- run `transforms::run_pipeline()`
- answer `query_model()`
- create remote contexts

### `ov::gfx_plugin::CompiledModel`
Implemented in `src/plugin/compiled_model.cpp`.

Responsibilities:
- own the transformed runtime model and original model
- create a backend state via `create_backend_state()`
- build the execution pipeline in `build_op_pipeline()`
- expose compiled-model properties and profiling state

The compiled pipeline is represented as `PipelineStageDesc` entries that wrap backend-specific `GpuStage` objects.

### `ov::gfx_plugin::InferRequest`
Declared in `include/openvino/gfx_plugin/infer_request.hpp` and implemented by backend-specific infer paths.

Responsibilities:
- bind host tensors or remote tensors
- create per-request backend state
- wire the compiled stage pipeline to actual buffers
- execute the pipeline and collect profiling data

## Pipeline Model
The runtime is stage-based.

`src/runtime/gpu_stage.hpp` defines the backend-neutral execution interface. Each backend registers a stage factory through `GpuStageFactory` in `src/runtime/execution_dispatcher.*`. During compilation, the plugin translates OpenVINO nodes into stage descriptors. During inference, these descriptors are cloned, connected to inputs and outputs, and executed in order.

The pipeline may fuse multiple nodes into a `FusedSequenceStage` when fusion logic decides that a combined execution path is valid.

## MLIR Role
MLIR lives in `src/mlir/` and is shared infrastructure, not a separate monolithic backend object.

It is used for:
- support probing through `mlir_supports_node()`
- node lowering via `mlir_builder_*.cpp`
- backend code generation helpers such as MSL or SPIR-V preparation

## Metal Backend
`src/backends/metal/` contains:
- `plugin/`: backend state creation, infer request, remote context, remote tensor
- `runtime/`: stage implementations, memory allocators, executor, profiler
- `codegen/`: Metal shader compilation helpers

Direct Metal API usage lives in Objective-C++ files (`.mm`).

## Vulkan Backend
`src/backends/vulkan/` mirrors the same broad split:
- `plugin/`: backend state, infer request, remote context, remote tensor
- `runtime/`: executor, buffers, profiling, runtime helpers
- `codegen/`: SPIR-V / Vulkan codegen helpers

This backend is built only when Vulkan support is available and enabled in CMake.

## Remote Contexts And Tensors
The shared abstractions are:
- `src/runtime/gfx_remote_context.*`
- `src/runtime/gfx_remote_tensor.*`

Backend-specific files provide the actual buffer wrapping and device-handle logic. Remote objects exist today, but the feature surface is still backend-dependent and not yet a fully polished portability layer.

## Import / Export Semantics
- `import_model()` reads a serialized OpenVINO model and recompiles it for the resolved backend.
- `export_model()` writes an OpenVINO model representation.

This is model serialization, not compiled-kernel caching.

## Important Constraints
- no partial CPU fallback
- backend parity is not guaranteed
- many ops still require static rank, static shape, or constant attributes
- the plugin remains experimental

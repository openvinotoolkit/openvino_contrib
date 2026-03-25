# Usage Guide

This document shows how to use the GFX plugin from an OpenVINO Runtime application.

## Registering A Local Build
For a plugin library built directly from this module, explicit registration is the most reliable workflow:

```cpp
#include <openvino/openvino.hpp>

ov::Core core;
core.register_plugin("/path/to/libopenvino_gfx_plugin.so", "GFX");
```

If the plugin is packaged into an OpenVINO deployment with plugin discovery metadata, explicit registration may not be required. For development builds, use `register_plugin()`.

## Reading Device Properties
```cpp
std::string backend = core.get_property("GFX", "GFX_BACKEND").as<std::string>();
std::string full_name = core.get_property("GFX", ov::device::full_name).as<std::string>();
auto capabilities = core.get_property("GFX", ov::device::capabilities);
```

Useful properties include:
- `GFX_BACKEND`
- `GFX_ENABLE_FUSION`
- `GFX_PROFILING_LEVEL`
- `GFX_PROFILING_REPORT`
- `GFX_MEM_STATS`
- `ov::device::id`
- `ov::enable_profiling`
- `PERF_COUNT`

## Selecting The Backend
The backend can be selected globally for the device:

```cpp
core.set_property("GFX", {{"GFX_BACKEND", "metal"}});
```

Or per compilation request:

```cpp
ov::CompiledModel compiled = core.compile_model(
    model,
    "GFX",
    {{"GFX_BACKEND", "vulkan"}});
```

The compiled model reports the resolved backend:

```cpp
std::string actual_backend = compiled.get_property("GFX_BACKEND").as<std::string>();
```

The selected backend affects both support probing and runtime route selection. In particular, Vulkan may enable specialized direct or chunked execution paths that are not available on Metal for the same graph.

## Compiling A Model
```cpp
auto model = core.read_model("/path/to/model.xml");

ov::AnyMap config = {
    {"GFX_BACKEND", "metal"},
    {"GFX_ENABLE_FUSION", true},
    {ov::enable_profiling.name(), true},
};

ov::CompiledModel compiled = core.compile_model(model, "GFX", config);
```

If the requested backend is unavailable on the current build or platform, `compile_model()` throws.

## Running Inference
```cpp
ov::InferRequest request = compiled.create_infer_request();
request.set_input_tensor(input_tensor);
request.infer();
ov::Tensor output = request.get_output_tensor();
```

Standard OpenVINO infer-request APIs for indexed or named inputs and outputs apply as usual.

## Profiling And Memory Statistics
Compiled models expose profiling and memory-related properties:

```cpp
ov::CompiledModel compiled = core.compile_model(
    model,
    "GFX",
    {
        {ov::enable_profiling.name(), true},
        {"GFX_PROFILING_LEVEL", 1},
    });

auto report = compiled.get_property("GFX_PROFILING_REPORT");
auto mem_stats = compiled.get_property("GFX_MEM_STATS");
```

The exact type returned by `GFX_MEM_STATS` depends on the active backend implementation and the headers visible to the caller.

Current runtime implementations also reuse some immutable device resources internally, such as constant buffers or prepared kernel bindings. These caches are internal optimization layers and do not require extra user API calls.

For static output signatures, infer requests may also reuse internal host output tensors when the application does not bind explicit output storage. If the application sets its own output tensor with `set_tensor()`, that user-provided tensor remains authoritative.

## Remote Contexts And Tensors
The plugin implements remote context and remote tensor interfaces, but effective capabilities remain backend-specific. If your integration depends on remote memory workflows, inspect:
- `src/runtime/gfx_remote_context.*`
- `src/runtime/gfx_remote_tensor.*`
- the active backend implementation under `src/backends/metal/plugin/` or `src/backends/vulkan/plugin/`

## Error Model
The plugin intentionally rejects unsupported models during compilation:
- there is no silent partial CPU fallback
- backend-specific capability differences surface through exceptions
- `query_model()` and `compile_model()` follow aligned support logic

For deployment, treat backend acceptance as backend-specific: a model accepted on one backend is not automatically guaranteed to compile on the other if a required optimized route, shape restriction, or device capability is missing.

For architectural and contributor details, continue with:
- `../README.md`
- `ARCHITECTURE.md`
- `DEVELOPMENT.md`

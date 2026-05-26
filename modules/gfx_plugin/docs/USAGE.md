# Usage Guide

This guide covers the public runtime surface of the `GFX` plugin.

## Registering The Plugin

When the plugin is not discovered through OpenVINO's plugin registry, register
the built library explicitly:

```cpp
#include <openvino/openvino.hpp>

ov::Core core;
core.register_plugin("/path/to/libopenvino_gfx_plugin.so", "GFX");
auto model = core.read_model("/path/to/model.xml");
auto compiled = core.compile_model(model, "GFX");
auto request = compiled.create_infer_request();
```

Python follows the normal OpenVINO flow:

```python
import openvino as ov

core = ov.Core()
core.register_plugin("/path/to/libopenvino_gfx_plugin.so", "GFX")
model = core.read_model("/path/to/model.xml")
compiled = core.compile_model(model, "GFX")
request = compiled.create_infer_request()
```

## Backend Selection

Use `GFX_BACKEND` to request a runtime backend:

```cpp
auto compiled = core.compile_model(model, "GFX", {
    {"GFX_BACKEND", "metal"},
});
```

Supported values:

- `metal`
- `opencl`

If `GFX_BACKEND` is omitted, the configured default from
`GFX_DEFAULT_BACKEND=auto|metal|opencl` is used. On macOS the available route is
Metal. On non-Apple builds, OpenCL is the source-kernel route when it is enabled
in the build.

The selected backend affects support probing, transforms, MLIR/source planning,
runtime binding, profiling counters, and which backend-specific infer request is
created.

## Public Properties

GFX-specific property names are declared in
`include/openvino/gfx_plugin/properties.hpp`.

Common properties:

- `GFX_BACKEND`: requested backend, `metal` or `opencl`
- `GFX_ENABLE_FUSION`: enable or disable plugin-level graph/stage fusion
- `GFX_PROFILING_LEVEL`: profiling detail level
- `GFX_PROFILING_REPORT`: read-only JSON profiling report on compiled models
- `GFX_MEM_STATS`: read-only memory statistics on compiled models
- `GFX_DIAGNOSTIC_F32_MPS_IMAGE`: Metal diagnostic route for selected f32
  MPS image placement checks

The plugin also exposes standard OpenVINO properties such as:

- `ov::available_devices`
- `ov::device::id`
- `ov::device::full_name`
- `ov::device::architecture`
- `ov::device::capabilities`
- `ov::execution_devices`
- `ov::hint::inference_precision`
- `ov::enable_profiling`
- `ov::cache_dir`

`ov::cache_dir` participates in normal OpenVINO model caching behavior. The
current GFX export path serializes the OpenVINO model, not a native backend
pipeline cache.

## Query And Compile

`query_model()` and `compile_model()` use the same backend-aware support rules.
If a model contains unsupported ops, shapes, or element types for the selected
backend, compilation should fail clearly.

```cpp
auto supported = core.query_model(model, "GFX", {
    {"GFX_BACKEND", "opencl"},
});
```

There is no partial CPU fallback for unsupported stages.

## Inference Precision

Use standard OpenVINO `ov::hint::inference_precision` when precision selection
is needed:

```cpp
auto compiled = core.compile_model(model, "GFX", {
    {ov::hint::inference_precision.name(), ov::element::f16},
});
```

Tests and compare tooling use precision-aware tolerances. Do not treat an FP16
route as if it must satisfy FP32-only thresholds unless a test explicitly asks
for strict thresholds.

## Profiling

Enable standard profiling and request a GFX profiling level:

```cpp
auto compiled = core.compile_model(model, "GFX", {
    {ov::enable_profiling.name(), true},
    {"GFX_PROFILING_LEVEL", "detailed"},
});

auto request = compiled.create_infer_request();
request.infer();
auto report = compiled.get_property("GFX_PROFILING_REPORT").as<std::string>();
```

The report contains compile and infer sections, target-profile fields, backend
counters, stage estimates, and backend-specific timing/counter data when
available. Confirm `extended.target_profile` or counters such as
`target_backend_metal` and `target_backend_opencl` before comparing runs.

## Remote Context And Remote Tensor

The public types are implemented as:

- `ov::gfx_plugin::GfxRemoteContext`
- `ov::gfx_plugin::GfxRemoteTensor`

Remote tensor support is backend-dependent. Use the public keys from
`properties.hpp` when passing backend buffers:

- `GFX_BUFFER`
- `GFX_MEMORY`
- `GFX_BUFFER_BYTES`
- `GFX_HOST_VISIBLE`
- `GFX_STORAGE_MODE`

The backend validates handle ownership, storage mode, size, and visibility
before binding.

## OpenCL Source Backend

The OpenCL backend dynamically loads the target OpenCL runtime and executes
source artifacts described by `src/kernel_ir/gfx_opencl_source_artifacts.*`.

Current public coverage includes selected data movement, shape/list movement,
Range/Tile, MatMul/Softmax, gather/scatter families, Concat/Split, typed
elementwise families, compare/select, and boolean logical/reduction families
when the model matches the artifact contracts.

Unsupported OpenCL cases fail during support probing, compilation, stage
creation, or runtime validation. They do not fall back to CPU or switch backend.

## Metal Backend

The Metal backend may select Apple MPS/MPSGraph vendor stages or Apple MSL
custom-kernel stages through shared stage policy. The request path uses explicit
MPSRT resource records, storage bridges, prepared resources, and kernel-buffer
orders when a typed MPSRT plan is present.

`GFX_DIAGNOSTIC_F32_MPS_IMAGE` is a diagnostic compile property for selected
f32 MPS image placement checks. It should be used for localization, not as a
general runtime switch.

## Compare Runner

`ov_gfx_compare_runner` is the preferred correctness triage tool:

```bash
ov_gfx_compare_runner \
  --model /path/to/model.xml \
  --device GFX \
  --reference-device TEMPLATE \
  --gfx-backend metal
```

Use it for numeric diffs, per-op narrowing, boolean output checks,
golden-reference comparisons, and real-image PPM inputs. Use `benchmark_app` or
`ov_gfx_microbench` for performance work.

## Microbench

`ov_gfx_microbench` supports `auto`, `metal`, and `opencl`:

```bash
ov_gfx_microbench --backend opencl --warmup 1 --iterations 3 \
  --output /tmp/gfx-microbench.json \
  --calibration-output /tmp/gfx-calibration.json
```

See `docs/MICROBENCH_SCHEMA.md` for the JSON contract and
`docs/PROFILING_RUNBOOK.md` for platform profiling flows.

## Known Constraints

- Backend parity is not guaranteed between Metal and OpenCL.
- Many ops require static rank, static shape, or constant attributes.
- OpenCL source artifacts intentionally cover a bounded set of role contracts.
- Metal vendor stages require strict descriptor, storage, and resource-binding
  contracts.
- Unsupported cases should fail clearly rather than taking hidden CPU or
  alternate backend routes.

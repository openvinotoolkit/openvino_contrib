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
in the build. Explicit default-backend requests are strict configure-time
requirements; CMake fails when the requested backend is unavailable.

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

Internally the selected backend is compiled through an in-memory compiler
service that builds a lowering plan, manifest, executable bundle, and runtime
descriptor. That descriptor is not a public cache format and is not exported by
`export_model()`.

The compiler registry contains the production backend compiler modules available
in the configured build, while runtime state creation is checked through the
configured backend support and runtime-provider registration. Requesting a
backend that is not supported by the current build fails during query or
compilation instead of falling through to another backend.

Compilation requires explicit backend routes. Removed generic routes such as
`backend_lowering` or `metal_lowering` are not public fallback paths; unsupported
operations fail during support probing or compilation.

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

Optional trace exporters are registered by backend code through the shared trace
sink registry. The Metal backend currently registers `signpost` and
`os_signpost` for `OV_GFX_PROFILE_TRACE`.

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
Those source artifacts are materialized by the OpenCL backend module, not by a
generic compiler fallback.

Current public coverage includes selected data movement, shape/list movement,
Range/Tile, MatMul/Softmax, Pool2D, bounded static NCHW spatial Interpolate,
gather/scatter families, Concat/Split, typed elementwise families,
compare/select, and boolean logical/reduction families when the model matches
the artifact contracts.

Generated activation, elementwise, f32 MatMul, f32/f16 Interpolate, f32
reduction, boolean reduction, f32/f16 Softmax, dynamic-static-rank f32/f16
Softmax, f32/f16 Pool2D, ShapeOf, Tile, Transpose, logical-bool elementwise,
compare/select, and Concat/Split helper sources are embedded under
`src/kernel_ir/opencl_kernels/`. The current Transpose route is
`opencl/generated/transpose_f32`; there is no active handwritten OpenCL
kernel-unit exception in the current registry. Interpolate is limited to f32/f16
static NCHW spatial resize cases with supported modes, axes, padding,
coordinate transforms, and nearest-rounding metadata. Pool2D is limited to
f32/f16 static 4D NCHW MaxPool/AvgPool contracts with 2D kernel, stride,
dilation, and padding metadata. OpenCL operation support requires a matching
source artifact and registered kernel unit; unsupported variants fail during
support probing or compilation.

Reduction source artifacts require static shape metadata and constant axes.
Numeric `ReduceSum`, `ReduceMean`, `ReduceMax`, `ReduceMin`, `ReduceProd`,
`ReduceL1`, and `ReduceL2` currently use the f32 generated source unit. Boolean
`ReduceLogicalAnd` and `ReduceLogicalOr` use the generated boolean reduction
source artifact.

For generated activation artifacts, `Swish` supports default beta, scalar
constant beta, and runtime scalar beta tensor forms when the beta input is a
static scalar tensor with the same element type as the data input. Other beta
shapes or element types are rejected by the artifact contract.

OpenCL `Softmax` supports f32/f16 static shapes and dynamic-output shapes with
static rank. The dynamic route carries runtime input-shape scalars in the
artifact ABI. OpenCL `LogSoftmax` is currently not implemented.

On Linux/Raspberry-style deployments, `GFX_ENABLE_RASPBERRY_OPENCL_TOOLCHAIN`
can build a plugin-local CLVK/CLSPV OpenCL bundle. The OpenCL loader checks the
plugin-local `opencl/`, `clvk/`, and plugin directories before system/vendor
OpenCL libraries. `src/backends/opencl/runtime/opencl_runtime_bundle.*`
describes plugin-adjacent bundles and configures bundled CLVK tool paths only
when the caller has not set them explicitly.

Unsupported OpenCL cases fail during support probing, compilation, stage
creation, or runtime validation. They do not fall back to CPU or switch backend.

## Metal Backend

The Metal backend may select Apple MPS/MPSGraph vendor primitive routes or
Apple MSL custom-kernel stages through shared stage policy. The request path
uses explicit MPSRT resource records, storage bridges, prepared resources, and
kernel-buffer orders when a typed MPSRT plan is present.

Compiler-owned Metal descriptors can also carry `VendorDescriptor` payloads for
supported MPS/MPSGraph primitives such as MatMul/GEMM, Softmax, Pool2D,
Resize2D, and SDPA. Those payloads are executed through the
`MpsrtVendorPrimitive` runtime stage only when the typed descriptor and
external-buffer ABI are valid. Generated Metal activation and elementwise
sources are planned through compiler-owned MSL descriptors rather than
request-time node checks. Generated Metal `Swish` activation follows the same
static-beta or runtime scalar-beta contract as the shared MLIR lowering.
Generated Metal reduction sources use the `metal/generated/reduction_f32` and
`metal/generated/reduction_logical_bool` contracts for the currently supported
f32 numeric and boolean logical forms. Generated Metal Softmax sources use
`metal/generated/softmax_f32`, `metal/generated/softmax_f16`,
`metal/generated/logsoftmax_f32`, and `metal/generated/logsoftmax_f16` for
static-shape f32/f16 Softmax and LogSoftmax. Static f32 Transpose can route
through `metal/generated/transpose_f32` when its shape and permutation contract
is satisfied. Metal Pool2D requires a valid MPS vendor descriptor; the generic
MSL Pool2D fallback is not a current route.

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

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
auto available_devices = core.get_property("GFX", ov::available_devices);
std::string selected_device_id = core.get_property("GFX", ov::device::id).as<std::string>();
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
- `ov::available_devices`
- `ov::device::id`
- `ov::cache_dir`
- `ov::enable_profiling`
- `PERF_COUNT`

`ov::available_devices` now exposes stable numeric ids such as `"0"`. Use
`ov::device::full_name` when you need the human-readable backend device name.

## Selecting The Backend
The backend can be selected globally for the device:

```cpp
core.set_property("GFX", {{"GFX_BACKEND", "metal"}});
```

Device selection is separate from backend selection:

```cpp
core.set_property("GFX", {{ov::device::id.name(), "0"}});
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

`GFX_PROFILING_REPORT` returns a JSON string. When profiling is enabled, the report may contain:
- a `compile` section with compile-time stage creation and compilation timings
  compile may also include kernel-cache and backend compiler subphases such as MSL/SPIR-V resolution, backend compilation, and shader/pipeline creation
- a node-level `nodes` section for standard OpenVINO profiling output
- an `extended` section with backend/runtime segments, counters, transfer totals, roofline heuristics, and diagnostics

`GFX_PROFILING_LEVEL` controls the amount of collected data:
- `0`: off
- `1`: standard infer and node-level profiling
- `2`: detailed profiling with extended runtime segments and counters

The profiling fast path stays disabled when profiling is off. The plugin does not collect runtime timestamps or extended counters unless profiling is explicitly enabled.

### Optional Trace Sinks
Detailed profiling can also expose an optional trace sink for external timeline analysis.

Current implementation uses the `OV_GFX_PROFILE_TRACE` environment variable:
- `perfetto` or `trace_event`: adds a `traceEvents` array to `GFX_PROFILING_REPORT`
- `signpost` or `os_signpost`: emits live `os_signpost` events on macOS

If `OV_GFX_PROFILE_TRACE_FILE` is set together with `OV_GFX_PROFILE_TRACE=perfetto`, the plugin also writes a merged Perfetto-compatible trace file that combines available compile and infer `traceEvents`.

The trace payload may include:
- segment spans for compile, submit, wait, upload, infer, download, and backend-specific hazard phases
- transfer spans for H2D and D2H activity
- allocation spans when allocation profiling is enabled
- backend binding/setup spans such as descriptor pool/create-write activity on Vulkan and pipeline/buffer binding on Metal
- final counter snapshots as Perfetto counter events

The extended JSON summary also includes lightweight roofline-style heuristics derived from recorded `flops_est`, `macs_est`, `bytes_in`, and `bytes_out`:
- per-phase arithmetic intensity and estimated achieved TFLOPS / GB/s
- an infer-level `roofline` aggregate
- a heuristic `dominant_regime` of `memory`, `mixed`, `compute`, or `unknown`

These values are inferential rather than hardware-calibrated peaks; they are intended for quick triage, not as a substitute for AGI, Perfetto, or vendor counters.

Example:

```bash
OV_GFX_PROFILE_TRACE=perfetto \
OV_GFX_PROFILE_TRACE_FILE=/tmp/gfx-trace.json \
./your_app
```

This sink selection is currently an internal activation path. The plugin does not expose a separate public property for trace sink selection.

### Vulkan Pipeline Cache Persistence
On Vulkan, the plugin can reuse the standard OpenVINO cache directory for backend pipeline-cache persistence:

```cpp
ov::CompiledModel compiled = core.compile_model(
    model,
    "GFX",
    {
        {"GFX_BACKEND", "vulkan"},
        {ov::cache_dir.name(), "/path/to/cache_dir"},
    });
```

This does not change `export_model()` semantics. It only gives the Vulkan backend a stable directory for native pipeline-cache files.

### Microbench Suite
`ov_gfx_microbench` provides the `MB0` to `MB3` suite used by the local profiling workflow docs.

- `MB0`: raw backend empty submit or empty Metal command-buffer commit with sync
- `MB1`: synthetic FP16 `Relu` model as a copy+dispatch approximation
- `MB2`: synthetic FP16 `Add` model as a bandwidth-pressure approximation
- `MB3`: synthetic FP16 `MatMul` model as a compute-pressure approximation

Example:

```bash
./ov_gfx_microbench \
  --backend metal \
  --warmup 3 \
  --iterations 10 \
  --output gfx-microbench.json \
  --calibration-output gfx-calibration.json
```

Assumptions:
- `MB1` to `MB3` reuse the normal GFX plugin path with `ov::enable_profiling=true` and `GFX_PROFILING_LEVEL=2`
- `MB1` to `MB3` are synthetic OpenVINO graphs, not backend-specific handwritten kernels
- each synthetic benchmark embeds the raw `GFX_PROFILING_REPORT` JSON so the same diagnostics pipeline can be reused on macOS, Android, and Raspberry Pi

The microbench JSON also includes a derived `analysis` section plus per-benchmark `workload`, `derived`, and `profile_digest` blocks:
- `device_key = vendor_id:device_id:driver_version` for device-keyed autotuning caches
- `fixed_overhead_us` from `MB0`
- `bandwidth_estimate_gbps` from `MB2`
- `compute_estimate_tflops` from `MB3`
- per-benchmark overhead-subtracted throughput estimates and lightweight hints such as `sync_heavy`, `transfer_heavy`, `binding_or_descriptor_churn`, or `compute_pressure_candidate`

These values are explicit heuristics from the synthetic suite. They are intended for quick per-device triage and autotuning seeds, not as calibrated peak hardware measurements.

For the complete contract and the reduced calibration artifact format, see:
- `MICROBENCH_SCHEMA.md`

For the cross-device profiling workflow, external trace capture, validation layers, and `perf` commands, see:
- `PROFILING_RUNBOOK.md`

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

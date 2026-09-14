# Profiling Runbook

This runbook covers GFX profiling on the current Metal and OpenCL backend
routes.

## What To Collect

For a useful profiling pass, collect:

- the exact build target and commit
- selected backend (`auto`, `metal`, or `opencl`)
- actual backend from compiled-model `GFX_BACKEND`
- `GFX_PROFILING_REPORT`
- `ov_gfx_microbench` output
- optional calibration artifact
- platform trace when investigating driver or GPU timing

Confirm route selection first. Reports may include `extended.target_profile` and
counters such as `target_backend_metal` and `target_backend_opencl`.

## Quick Microbench

```bash
ov_gfx_microbench --backend metal --warmup 1 --iterations 3 \
  --output /tmp/gfx-microbench-metal.json \
  --calibration-output /tmp/gfx-calibration-metal.json
```

```bash
ov_gfx_microbench --backend opencl --warmup 1 --iterations 3 \
  --output /tmp/gfx-microbench-opencl.json \
  --calibration-output /tmp/gfx-calibration-opencl.json
```

Use `docs/MICROBENCH_SCHEMA.md` for field definitions.

## Generate Platform Commands

`tools/gfx_profile_runbook.py` prints ready-to-run command sets:

```bash
python3 tools/gfx_profile_runbook.py --platform macos --format shell
python3 tools/gfx_profile_runbook.py --platform android --format shell
python3 tools/gfx_profile_runbook.py --platform rpi --format shell
```

The generated Android and Raspberry Pi commands use the OpenCL backend. Override
paths with the script options when using a non-default deployment directory.

## GFX Profiling Report

Enable profiling when compiling the model:

```cpp
auto compiled = core.compile_model(model, "GFX", {
    {ov::enable_profiling.name(), true},
    {"GFX_PROFILING_LEVEL", "detailed"},
});
```

After inference:

```cpp
auto report = compiled.get_property("GFX_PROFILING_REPORT").as<std::string>();
```

Important report areas:

- `compile`: compile-time stages and backend source/pipeline preparation
- `infer`: request execution, copies, waits, submits, and stage execution
- `extended.target_profile`: backend, device family, workgroup limits, memory
  alignment, and capability flags
- counters: backend route, dispatch count, pipeline creation, descriptor/binding
  work, resource allocation, and route-specific MPSRT/OpenCL counters

Trace export is selected by `OV_GFX_PROFILE_TRACE`. Trace sinks are registered
by backend code through `src/runtime/gfx_profiling_trace_sink.*`; the Metal
backend currently registers `signpost` and `os_signpost`.

## Common Bottleneck Categories

### Fixed Overhead

Look at:

- MB0 median wall time
- `fixed_overhead_us`
- `fixed_overhead_share`
- first-to-steady ratios

High fixed overhead matters most for tiny models and many small dispatches.

### Synchronization

Look at:

- wait segments
- `wait_share_of_wall`
- `final_fence_wait_seen`
- submit and barrier counts
- dependency-window extension counters

Check whether submission windows are split by required boundaries such as
layout, split, transpose, softmax, or attention stages.

### Transfers

Look at:

- H2D/D2H bytes
- upload/download segments
- `transfer_share_of_wall`
- host output reuse behavior
- remote tensor binding path

Unexpected transfer pressure usually means a tensor is being materialized or
copied when a backend buffer or view-style alias should have been reused.

### Binding Or Descriptor Churn

Look at:

- descriptor or binding update counters
- prepared binding cache behavior
- `binding_prepare_in_infer`
- Metal `mpsrt_*` resource-binding counters
- OpenCL program or kernel setup spans

Repeated binding setup inside steady-state inference usually points to an
incomplete cache key, changing runtime shape, or missing prepared resource.

### Compile-In-Infer

Look at:

- pipeline creation counters
- first infer versus median infer time
- backend compiler spans
- OpenCL program cache misses
- Metal MSL/MPSRT prepared-pipeline cache counters

Pipeline creation should not move into steady-state inference unless the runtime
shape or backend contract requires a new compiled variant.

## Platform Notes

### macOS / Metal

Use:

- Instruments or `xcrun xctrace`
- Xcode GPU capture when command-buffer contents need inspection
- signpost or Perfetto-style trace export when correlating CPU and plugin spans

Example:

```bash
OV_GFX_PROFILE_TRACE=signpost \
OV_GFX_PROFILE_TRACE_FILE=/tmp/gfx-trace-macos.json \
benchmark_app -m /path/to/model.xml -d GFX -pc -niter 10
```

`signpost` and `os_signpost` are Metal-registered trace sink names. They are
available only when the Metal backend code is present and has registered its
profiling sink.

For Metal placement issues, capture the GFX profiling report together with
`ov_gfx_compare_runner --dump-gfx-profile --gfx-profiling-level detailed` so
accuracy evidence and placement counters stay tied to the same run.

### Android / OpenCL

Use:

- Perfetto or AGI for device-side timeline evidence
- OpenCL profiling counters when available
- the generated `tools/gfx_profile_runbook.py --platform android` commands

Deploy the plugin, OpenVINO runtime libraries, model, and `ov_gfx_microbench`
into the same remote directory. Confirm that the target has a working OpenCL GPU
runtime before interpreting backend performance.

### Raspberry Pi / OpenCL

Use:

- `perf stat` for coarse CPU counters
- `perf record` for call-stack attribution
- OpenCL route counters from the GFX profiling report

Treat Raspberry Pi measurements as target-specific. Compare against another run
only when backend, device key, model, runtime libraries, and calibration
artifact match.

## Trace And Artifact Helpers

- `tools/gfx_microbench_smoke.py`: validates microbench/calibration round trips
- `tools/gfx_calibration_diff.py`: compares calibration artifacts
- `tools/gfx_external_trace_summary.py`: summarizes external trace exports
- `tools/gfx_profile_runbook.py`: prints command bundles for platform runs

## Triage Order

1. Confirm actual backend and target profile.
2. Check correctness with `ov_gfx_compare_runner` if output quality is in doubt.
3. Run `ov_gfx_microbench` and capture calibration.
4. Read the GFX profiling report for submit, wait, transfer, compile, and
   binding counters.
5. Capture a platform trace only after plugin counters identify the likely
   category.
6. Fix the shared planner, manifest, artifact, or backend boundary that owns the
   observed behavior.

## Reporting Results

When reporting profiling work, include:

- exact command lines
- backend requested and backend actually used
- key report counters
- whether calibration matched
- platform trace tool used, if any
- conclusion category: fixed overhead, synchronization, transfer, binding churn,
  compile-in-infer, or compute pressure
- next measurement or fix target

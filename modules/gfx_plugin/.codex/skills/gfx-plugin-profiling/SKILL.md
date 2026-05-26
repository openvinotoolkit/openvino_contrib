---
name: gfx-plugin-profiling
description: Use when analyzing GFX plugin performance, profiling reports, microbench output, trace correlation, or backend overhead on macOS, Android, Linux, or Raspberry Pi for Metal and OpenCL paths.
---

# GFX Plugin Profiling

This skill is for performance triage and profiling workflows in
`modules/gfx_plugin/`.

## Use This Skill When

- The user asks about performance, bottlenecks, slow inference, submit overhead,
  cache effects, transfer pressure, or synchronization cost.
- The task mentions `GFX_PROFILING_REPORT`, `ov_gfx_microbench`, MB0-MB3,
  Perfetto, AGI, Xcode/Instruments, or `perf`.
- The task needs interpretation of profiling JSON, calibration artifacts, or
  trace output.
- The task compares `metal`, `opencl`, or `auto` and needs to confirm which
  backend route actually ran.

## Read First

1. `docs/PROFILING_RUNBOOK.md`
2. `docs/MICROBENCH_SCHEMA.md`
3. `docs/USAGE.md`
4. `docs/TESTING.md`
5. `README.md`

## Tooling Surface

- `ov_gfx_microbench`
- `GFX_PROFILING_REPORT`
- `tools/gfx_profile_runbook.py`
- `tools/gfx_microbench_smoke.py`
- `tools/gfx_calibration_diff.py`
- `tools/gfx_external_trace_summary.py`

Native trace surfaces:

- macOS: Instruments, `xcrun xctrace`, Xcode GPU capture, signposts
- Android: Perfetto or AGI
- Linux/Raspberry Pi: `perf stat`, `perf record`

## Core Workflow

1. Rebuild the relevant binary.
2. Run `ov_gfx_microbench` and save report plus calibration JSON.
3. Run the real workload with profiling enabled.
4. Capture a platform trace only when plugin counters point to a category.
5. Correlate:
   - `analysis.triage_hints`
   - `benchmarks[].derived`
   - `benchmarks[].profile_digest`
   - `GFX_PROFILING_REPORT.compile`
   - `GFX_PROFILING_REPORT.extended`
   - `GFX_PROFILING_REPORT.extended.target_profile`

## Inspect First

### Fixed Overhead

- MB0 fixed overhead
- `fixed_overhead_us`
- `fixed_overhead_share`
- first-to-steady ratios

### Synchronization

- wait segments
- `wait_share_of_wall`
- final fence wait counters
- submit and barrier counts
- dependency-window extension counters

### Transfers

- H2D/D2H bytes
- upload/download spans
- `transfer_share_of_wall`
- remote tensor and output reuse paths

### Binding Or Descriptor Churn

- descriptor or binding update counters
- prepared binding cache behavior
- `binding_prepare_in_infer`
- Metal `mpsrt_*` resource-binding counters
- OpenCL program/kernel setup spans

### Compile Or Cache Regression

- `compile_ms`
- pipeline creation counters
- OpenCL program-cache behavior
- Metal MSL/MPSRT prepared-pipeline cache counters
- first-infer versus steady-state timing

## Interpretation Rules

- Treat microbench numbers as heuristics, not peak hardware claims.
- Use `ov_gfx_compare_runner` for correctness and `ov_gfx_microbench` for
  profiling triage.
- Distinguish wall time, GPU time, and overhead-subtracted estimates.
- Confirm `actual_backend`, `extended.target_profile`, or target backend
  counters before comparing `auto`, Metal, and OpenCL runs.
- Treat standalone OpenCL microbench output as kernel evidence only; plugin
  performance claims require execution through the GFX OpenCL backend.

## Platform Notes

### macOS

- Prefer signposts or `xctrace` when correlating CPU and Metal execution.
- Compare command-buffer timing with plugin stage/segment timing.
- Capture GFX profile output with compare-runner when placement counters and
  accuracy need to be tied to the same run.

### Android / OpenCL

- Use Perfetto or AGI for GPU busy/idle gaps and CPU blocking around queue
  submits or waits.
- Confirm the target has a working OpenCL GPU runtime before interpreting
  backend performance.

### Linux / Raspberry Pi OpenCL

- Use `perf stat` and `perf record` for CPU-side attribution.
- Keep backend, device key, model, runtime libraries, and calibration artifact
  matched before comparing runs.

## Output Expectations

- State the most likely bottleneck category first.
- Point to exact fields or tool outputs.
- Recommend the next measurement step or fix target.

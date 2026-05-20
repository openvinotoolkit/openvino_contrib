---
name: gfx-plugin-profiling
description: Use when analyzing GFX plugin performance, profiling reports, microbench output, trace correlation, or backend overhead on macOS, Android, or Raspberry Pi for Metal and Vulkan paths.
---

# GFX Plugin Profiling

This skill is for performance triage and profiling workflows in `modules/gfx_plugin/`.

## Use This Skill When

- The user asks about performance, bottlenecks, slow inference, submit overhead, cache effects, transfer pressure, or synchronization cost.
- The task mentions `GFX_PROFILING_REPORT`, `ov_gfx_microbench`, `MB0` to `MB3`, Perfetto, AGI, Xcode/Instruments, or `perf`.
- The task needs interpretation of profiling JSON, calibration artifacts, or trace-event output.

## Primary References

Read in this order:

1. `docs/PROFILING_RUNBOOK.md`
2. `docs/MICROBENCH_SCHEMA.md`
3. `docs/USAGE.md`
4. `docs/TESTING.md`
5. `README.md`

## Tooling Surface

Main tools and outputs:

- `ov_gfx_microbench`
- `GFX_PROFILING_REPORT`
- `tools/gfx_profile_runbook.py`
- `tools/gfx_microbench_smoke.py`
- `tools/gfx_calibration_diff.py`
- `tools/gfx_external_trace_summary.py`

Native trace surfaces by platform:

- macOS: `xctrace`, Instruments, Xcode GPU capture, signposts
- Android: AGI, Perfetto, validation layers
- Raspberry Pi: `perf stat`, `perf record`

## Core Workflow

1. Rebuild the relevant binary in the existing build directory.
2. Run `ov_gfx_microbench` and save:
   - full report JSON
   - calibration artifact JSON
3. Run the real workload with profiling enabled.
4. Capture the platform-native trace.
5. Correlate:
   - `analysis.triage_hints`
   - `benchmarks[].derived`
   - `benchmarks[].profile_digest`
   - `GFX_PROFILING_REPORT.compile`
   - `GFX_PROFILING_REPORT.extended`

## What To Inspect First

### Fixed overhead or sync suspicion

Check:

- `MB0.fixed_overhead_us`
- `wait_share_of_wall`
- `final_fence_wait_seen`
- `submit_count`
- `barrier_count`

### Transfer-heavy suspicion

Check:

- `bytes_in`, `bytes_out`, `bytes_moved`
- `transfer_share_of_wall`
- H2D/D2H transfer spans
- upload/download segments in profiling report

### Descriptor or binding churn suspicion

Check:

- `descriptor_update_count`
- `binding_prepare_in_infer`
- `submission_dependency_window_extension_count`
- `submission_dependency_extension_budget_num` / `submission_dependency_extension_budget_den`
- backend setup spans in trace output
- Metal MPSRT counters such as `mpsrt_*_bound_resource_count`, `mpsrt_mps_resize2d_*`, `mpsrt_mps_graph_*`, and `mpsrt_encode` segments when the shared `gfx_mpsrt_model.*` resource table, external-buffer binding plan, prepared heap, storage-bridge path, or MPSGraph-backed vendor route is in use

### Compile or cache regression suspicion

Check:

- `compile_ms`
- `pipeline_creation_count`
- MPSRT vendor-kernel cache counters such as `mpsrt_mps_resize2d_kernel_cache_hit_count` and `mpsrt_mps_resize2d_kernel_cache_miss_count`
- `ov::cache_dir` usage
- whether pipeline creation moved into infer-time spans

## Interpretation Rules

- Treat microbench numbers as heuristics, not calibrated peak hardware measurements.
- Use `ov_gfx_compare_runner` for correctness and `ov_gfx_microbench` for profiling triage; do not mix their purposes.
- Distinguish wall-time, GPU-time, and overhead-subtracted estimates.
- Correlate plugin-internal profiling with platform-native traces before concluding that a backend route is the bottleneck.
- Prefer `ov_gfx_compare_runner --dump-gfx-profile --gfx-profiling-level detailed` when accuracy triage and placement counters need to be captured in the same run; keep it accuracy-only and use `benchmark_app` or microbench tools for performance numbers.

## Platform Notes

### macOS

- Prefer signposts or Perfetto-style export when correlating CPU and Metal execution.
- Compare command-buffer timing with plugin segment timing.

### Android

- Use AGI or Perfetto for GPU busy/idle gaps and CPU blocking around queue submit or fence waits.
- Use validation layers only for correctness, not for performance numbers.

### Raspberry Pi

- Use `perf stat` and `perf record` for coarse CPU-side evidence.
- Correlate Broadcom/Vulkan driver overhead with plugin-side submit, wait, and transfer counters.

## Output Expectations

- State the most likely bottleneck category first: fixed overhead, sync, transfer, binding churn, compile-in-infer, or compute pressure.
- Point to the exact fields or tool outputs that support the conclusion.
- Recommend the next measurement step, not just a diagnosis.

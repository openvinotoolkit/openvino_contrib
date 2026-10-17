# Microbench Schema

`ov_gfx_microbench` emits JSON for fixed-overhead and small workload profiling.
The supported backend selector is `auto|metal|opencl`.

## Top-Level Report

The report has this shape:

```json
{
  "schema_version": 1,
  "selected_backend": "auto",
  "device": {
    "backend": "metal",
    "device_name": "Apple GPU",
    "full_name": "Apple GPU",
    "platform": "macOS",
    "vendor_id": "apple",
    "device_id": "metal_default",
    "driver_version": "metal",
    "architecture": "apple_silicon"
  },
  "benchmarks": [],
  "analysis": {}
}
```

Fields:

- `schema_version`: JSON contract version.
- `selected_backend`: requested CLI backend.
- `device.backend`: backend that the benchmark fingerprint represents.
- `benchmarks`: MB0-MB3 result entries.
- `analysis`: derived hints and calibration comparison output.

For OpenCL, device fields are populated from the dynamically loaded target
runtime when available. For Metal, the fingerprint uses the active Metal device
information available to the tool.

## Benchmark Entry

Typical benchmark entries include:

```json
{
  "name": "MB1",
  "kind": "plugin",
  "model": "single_element_add",
  "actual_backend": "metal",
  "compile_ms": 2.3,
  "first_infer_ms": 0.21,
  "median_infer_ms": 0.05,
  "min_infer_ms": 0.04,
  "max_infer_ms": 0.07,
  "profile": {},
  "derived": {}
}
```

Important fields:

- `actual_backend`: backend returned by compiled-model `GFX_BACKEND`.
- `compile_ms`: model compilation wall time for the benchmark.
- `first_infer_ms`: first request latency.
- `median_infer_ms`, `min_infer_ms`, `max_infer_ms`: steady-state latency
  summary after warmup.
- `profile`: parsed GFX profiling report when profiling data is available.
- `derived`: calculated ratios, throughput estimates, and triage hints.

## MB0

`MB0` measures the lowest-level empty backend submit or command-buffer overhead
available on the platform:

- on macOS, a Metal empty command-buffer commit/sync path
- on non-Apple builds, an OpenCL empty queue/finish path

The MB0 entry includes:

```json
{
  "name": "MB0",
  "backend": "opencl",
  "median_wall_us": 12.0,
  "min_wall_us": 10.0,
  "max_wall_us": 15.0,
  "median_gpu_us": 0.0,
  "has_gpu_us": false
}
```

Use MB0 as an overhead baseline, not as a hardware peak metric.

## Derived Fields

The `derived` object may include:

- `fixed_overhead_us`
- `fixed_overhead_share`
- `overhead_subtracted_ms`
- `e2e_tflops`
- `e2e_gbps`
- `adjusted_tflops`
- `adjusted_gbps`
- `gpu_tflops`
- `gpu_gbps`
- `gpu_share_of_wall`
- `wait_share_of_wall`
- `transfer_share_of_wall`
- `first_to_steady_ratio`
- `triage_hints`

These are heuristic profiling aids. Do not use them as correctness gates.

## Profile Digest

When a benchmark contains a GFX profiling report, the tool derives a compact
profile digest with fields such as:

- `has_extended`
- `has_compile`
- `counters_supported`
- `counters_used`
- `total_gpu_us`
- `total_cpu_us`
- `total_wall_us`
- `total_h2d_bytes`
- `total_d2h_bytes`
- `wait_cpu_us`
- `submit_cpu_us`
- `barrier_cpu_us`
- `upload_cpu_us`
- `download_cpu_us`
- `final_fence_wait_cpu_us`
- `submit_count`
- `barrier_count`
- `descriptor_update_count`
- `pipeline_creation_count`
- `total_node_dispatches`
- convolution planning counters when present

Always confirm `actual_backend`, `device.backend`, or target-profile counters
before comparing reports from different backend routes.

## Calibration Artifact

`--calibration-output` writes a compact device/backend baseline:

```json
{
  "schema_version": 1,
  "backend": "opencl",
  "device_key": "opencl_gpu",
  "mb0_fixed_overhead_us": 12.0,
  "created_utc": "2026-05-26T00:00:00Z"
}
```

The microbench loader compares a supplied calibration artifact against the
current run and records:

- `device_key_match`
- `backend_match`
- `schema_match`

Treat mismatches as a warning that fixed-overhead subtraction may not be
comparable.

## Example Commands

Metal:

```bash
ov_gfx_microbench --backend metal --warmup 1 --iterations 3 \
  --output /tmp/gfx-microbench-metal.json \
  --calibration-output /tmp/gfx-calibration-metal.json
```

OpenCL:

```bash
ov_gfx_microbench --backend opencl --warmup 1 --iterations 3 \
  --output /tmp/gfx-microbench-opencl.json \
  --calibration-output /tmp/gfx-calibration-opencl.json
```

Auto:

```bash
ov_gfx_microbench --backend auto --warmup 1 --iterations 3
```

## Interpretation Rules

- Compare `actual_backend` and calibration `backend` before using derived
  numbers.
- Compare device keys before subtracting fixed overhead from another run.
- Use `ov_gfx_compare_runner` for correctness. Microbench output is profiling
  evidence only.
- Backend-specific trace tools are still needed before assigning a bottleneck to
  CPU submission, GPU execution, transfers, or driver behavior.

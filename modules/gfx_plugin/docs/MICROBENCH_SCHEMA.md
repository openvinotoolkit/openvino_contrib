# GFX Microbench Schema

`ov_gfx_microbench` produces two JSON contracts:
- a full microbench report
- a reduced calibration artifact keyed by device identity

Both contracts are observational. They are intended for triage and autotuning seeds, not as calibrated peak-hardware measurements.

## Full Microbench Report

Current root contract:

```json
{
  "schema_version": 2,
  "tool": "ov_gfx_microbench",
  "selected_backend": "metal|vulkan|auto",
  "device": { "...": "..." },
  "warmup": 3,
  "iterations": 10,
  "assumptions": ["..."],
  "mb0": { "...": "..." },
  "analysis": { "...": "..." },
  "calibration": { "...": "..." },
  "loaded_calibration": { "...": "..." },
  "benchmarks": [{ "...": "..." }]
}
```

Root fields:
- `schema_version`: report schema version. Current value is `2`.
- `tool`: emitting tool. Current value is `ov_gfx_microbench`.
- `selected_backend`: requested backend selection.
- `device`: resolved device fingerprint.
- `warmup`, `iterations`: benchmark loop parameters.
- `assumptions`: explicit assumptions for `MB0` to `MB3`.
- `mb0`: raw empty-submit measurement.
- `analysis`: top-level derived metrics and triage hints.
- `calibration`: reduced persistent artifact derived from the current run.
- `loaded_calibration`: summary of an artifact loaded with `--calibration-input`.
- `benchmarks`: `MB1`, `MB2`, `MB3` entries.

### `device`

```json
{
  "backend": "metal|vulkan",
  "device_name": "Apple M1 Max",
  "full_name": "GFX (Apple M1 Max)",
  "platform": "macos|linux_or_android",
  "vendor_id": "0x5143|apple",
  "device_id": "0x44050001|metal_default",
  "driver_version": "2150760449|metal",
  "architecture": "vulkan|apple_silicon"
}
```

`device_key = vendor_id:device_id:driver_version` is used as the stable calibration key.

### `mb0`

`MB0` is the fixed-overhead probe described in `create-profiler.md`.

```json
{
  "backend": "metal|vulkan",
  "median_wall_us": 33.416,
  "min_wall_us": 31.002,
  "max_wall_us": 36.771,
  "median_gpu_us": null
}
```

### `analysis`

`analysis` is the top-level summary used for quick triage.

```json
{
  "device_key": "0x5143:0x44050001:2150760449",
  "fixed_overhead_us": 416.250,
  "bandwidth_probe": "MB2",
  "compute_probe": "MB3",
  "bandwidth_estimate_gbps": 14.123,
  "compute_estimate_tflops": 0.073,
  "gpu_bandwidth_estimate_gbps": 0.000,
  "gpu_compute_estimate_tflops": 0.075,
  "triage_hints": ["mb3_sync_bound"],
  "assumptions": ["..."]
}
```

Field meaning:
- `fixed_overhead_us`: `MB0.median_wall_us`.
- `bandwidth_estimate_gbps`: `MB2.bytes_moved / (MB2.median_infer_ms - MB0)`.
- `compute_estimate_tflops`: `MB3.flops_est / (MB3.median_infer_ms - MB0)`.
- `gpu_*`: same estimates but using `profile.extended.total_gpu_us` when available.
- `triage_hints`: coarse diagnostic flags for developer triage.

### `benchmark`

Each `MB1` to `MB3` entry contains:

```json
{
  "name": "MB3",
  "kind": "gemm_model",
  "model": "matmul_const<f16>[1x1024x1024]x[1x1024x1024]",
  "actual_backend": "vulkan",
  "compile_ms": 37.231,
  "first_infer_ms": 38.226,
  "median_infer_ms": 29.946,
  "min_infer_ms": 29.946,
  "max_infer_ms": 29.946,
  "workload": { "...": "..." },
  "derived": { "...": "..." },
  "profile_digest": { "...": "..." },
  "profile": { "...": "full GFX_PROFILING_REPORT ..." }
}
```

#### `workload`

Synthetic-workload estimate used for derived metrics.

```json
{
  "bytes_in": 4194304,
  "bytes_out": 2097152,
  "bytes_moved": 6291456,
  "macs_est": 1073741824,
  "flops_est": 2147483648,
  "arithmetic_intensity": 341.333,
  "note": "MatMul FLOP estimate assumes 2 FLOPs per MAC and excludes cache-reuse effects."
}
```

#### `derived`

Quick per-probe metrics:
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
- `hints`

#### `profile_digest`

Reduced summary extracted from `profile.extended`:
- total times and transfer volumes
- phase CPU slices for `wait`, `submit`, `barrier`, `upload`, `download`
- selected counters such as `submit_count`, `barrier_count`, `descriptor_update_count`, `pipeline_creation_count`
- booleans for `sync_heavy`, `transfer_heavy`, `compile_in_infer`, `binding_prepare_in_infer`, `final_fence_wait_seen`, `cross_submit_barrier_seen`

This digest exists to make calibration consumers independent from the full profiling JSON layout.

## Calibration Artifact

The persistent artifact is emitted with `--calibration-output`.

Current contract:

```json
{
  "schema_version": 1,
  "microbench_schema_version": 2,
  "tool": "ov_gfx_microbench",
  "device_key": "0x5143:0x44050001:2150760449",
  "backend": "vulkan",
  "device_name": "Adreno (TM) 830",
  "platform": "linux_or_android",
  "vendor_id": "0x5143",
  "device_id": "0x44050001",
  "driver_version": "2150760449",
  "fixed_overhead_us": 416.250,
  "bandwidth_estimate_gbps": 14.123,
  "compute_estimate_tflops": 0.073,
  "gpu_bandwidth_estimate_gbps": 0.000,
  "gpu_compute_estimate_tflops": 0.075,
  "triage_hints": ["mb3_sync_bound"],
  "assumptions": ["..."],
  "probes": [{ "...": "..." }]
}
```

`probes[]` contains one reduced entry per `MB1`, `MB2`, `MB3`:
- `name`
- `actual_backend`
- `arithmetic_intensity`
- `overhead_subtracted_ms`
- `adjusted_gbps`
- `adjusted_tflops`
- `gpu_gbps`
- `gpu_tflops`
- `first_to_steady_ratio`
- `wait_share_of_wall`
- `transfer_share_of_wall`
- `submit_count`
- `barrier_count`
- `hints`

## CLI

```bash
./ov_gfx_microbench \
  --backend metal \
  --warmup 3 \
  --iterations 10 \
  --output gfx-microbench.json \
  --calibration-output gfx-calibration.json
```

To load and compare an existing artifact:

```bash
./ov_gfx_microbench \
  --backend vulkan \
  --warmup 1 \
  --iterations 3 \
  --calibration-input gfx-calibration.json
```

When `--calibration-input` is used, the full report includes `loaded_calibration` with:
- `device_key_match`
- `backend_match`
- `schema_match`

## Assumptions

- `MB1` to `MB3` are synthetic FP16 graphs through the normal GFX plugin path.
- `MB1` approximates copy+dispatch, `MB2` approximates bandwidth pressure, `MB3` approximates compute pressure.
- `adjusted_*` metrics subtract only `MB0` median fixed overhead.
- `gpu_*` estimates depend on `extended.total_gpu_us`; if GPU counters are unavailable they may be `0`.

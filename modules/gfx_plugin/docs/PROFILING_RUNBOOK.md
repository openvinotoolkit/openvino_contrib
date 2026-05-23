# GFX Profiling Runbook

This runbook is the operational companion to the local profiling and microbench workflow docs in this module.

Use it when you need to:
- collect `GFX_PROFILING_REPORT`
- run `MB0` to `MB3`
- capture external traces and counters
- correlate tool output with AGI, Instruments, Xcode GPU capture, or `perf`

Automation helpers live in `modules/gfx_plugin/tools/`:
- `gfx_profile_runbook.py`: emits ready-to-run commands for macOS, Android, or Raspberry Pi
- `gfx_microbench_smoke.py`: verifies `ov_gfx_microbench` plus calibration roundtrip
- `gfx_calibration_diff.py`: produces a machine-readable diff between two calibration artifacts
- `gfx_external_trace_summary.py`: summarizes trace-event JSON, generic JSON exports, `perf stat`, or `perf report`

The helper script uses generic public defaults. Override them with command-line options or environment variables such as `GFX_MACOS_BUILD`, `GFX_ANDROID_BUILD`, `GFX_RPI_BUILD`, `GFX_MODEL`, `GFX_ANDROID_DIR`, and `GFX_RPI_DIR` when your local checkout uses different paths.

## Unified Flow

1. Rebuild the relevant target in the existing build directory.
2. Run `ov_gfx_microbench` and save:
   - full report JSON
   - calibration artifact JSON
3. Run the real workload with profiling enabled:
   - `benchmark_app -pc`
   - or the integration binary under test
4. Capture the platform-native trace:
   - Android: AGI / Perfetto and optional validation layers
   - macOS: `xctrace` and Xcode Metal workload capture
   - Raspberry Pi: `perf stat` and `perf record`
5. Correlate the native trace with:
   - `analysis.triage_hints`
   - `benchmarks[].derived`
   - `benchmarks[].profile_digest`
   - `GFX_PROFILING_REPORT.compile`
   - `GFX_PROFILING_REPORT.extended`
   - `GFX_PROFILING_REPORT.extended.target_profile`

Recent infer profiling also records lightweight per-stage estimates on `stage_execute` segments:
- `bytes_in`
- `bytes_out`
- `macs_est`
- `flops_est`

These are estimates, not hardware counters, but they are useful when comparing trace hot spots with the roofline-style summaries in the profiling report.

Recent backend profiling also records a target profile when the active buffer
manager reports device information. Check `extended.target_profile` and counters
such as `target_backend_opencl`, `target_backend_metal`, and
`target_backend_vulkan` before comparing runs from different backend routes.

If you want the script-generated commands instead of following the sections manually:

```bash
python3 modules/gfx_plugin/tools/gfx_profile_runbook.py --platform macos --format shell
python3 modules/gfx_plugin/tools/gfx_profile_runbook.py --platform android --format json
python3 modules/gfx_plugin/tools/gfx_profile_runbook.py --platform rpi --format shell
```

## Hazard Signals

Treat these as triage flags, not as automatic proof of a bug.

- `owns_command_buffer`
  Meaning: a Vulkan kernel executed outside the shared infer command buffer.
  Read from: `extended.summary.counter_map` and hazard segments in `GFX_PROFILING_REPORT`.
  Triage meaning: per-kernel submit/wait path may still be active.

- `final_fence_wait`
  Meaning: end-of-infer host wait on the submission fence.
  Read from: `benchmarks[].profile_digest.final_fence_wait_seen` and `wait_share_of_wall`.
  Triage meaning: infer is sync-heavy.

- `cross_submit_memory_barrier`
  Meaning: explicit cross-submit barrier was emitted.
  Read from: `barrier_count`, `cross_submit_barrier_seen`, segment names.
  Triage meaning: submit partitioning or dependency tracking may still be too coarse.

- `submission_dependency_window_extension_count`
  Meaning: a direct producer-consumer chain stayed in the current command-buffer window after a soft stage/output/MAC budget boundary.
  Read from: `extended.summary.counter_map`, together with `submission_dependency_extension_budget_num` and `submission_dependency_extension_budget_den`.
  Triage meaning: dependency-aware batching is active; if synchronization still dominates, inspect whether a boundary stage is forcing the next submit.

- `pipeline_creation_count`
  Meaning: pipeline or shader creation happened during infer.
  Read from: `profile_digest.pipeline_creation_count`.
  Triage meaning: prewarm or cache persistence is incomplete.

- `target_backend_opencl`, `target_backend_metal`, `target_backend_vulkan`
  Meaning: which backend route produced the profiled request.
  Read from: `extended.summary.counter_map` and `extended.target_profile`.
  Triage meaning: confirm that `auto` resolved to the intended backend before interpreting route-specific counters.

- `descriptor_update_count` or binding-preparation diagnostics
  Meaning: CPU work is spent rebinding descriptors or backend-specific binding tables.
  Read from: `profile_digest.descriptor_update_count`, `binding_prepare_in_infer`, diagnostics.
  Triage meaning: descriptor churn is visible.

- `mpsrt_mps_resize2d_*`, `mpsrt_mps_graph_*`, `mpsrt_*_bound_resource_count`, `mpsrt_prepared_resource_heap_*`, or prepared-resource diagnostics
  Meaning: Metal MPSRT work is spending CPU time preparing or binding Apple MPS vendor resources, image bridges, runtime-parameter buffers, or heap-backed transient resources.
  Read from: `extended.summary.counter_map`, `mpsrt_encode` segments, prepared-model counters such as `mpsrt_prepared_resource_heap_alias_reuse_count`, request counters such as `mpsrt_mps_graph_sdpa_request_encode_count`, and compile counters such as `mpsrt_prepare_mps_resize2d_count`.
  Triage meaning: the MPSRT resource table, storage bridges, or prepared heap path should be inspected before treating the kernel itself as the bottleneck.

## Microbench Commands

### macOS

```bash
mkdir -p .gfx-profile
/path/to/build-gfx-plugin-macos/output/bin/arm64/RelWithDebInfo/ov_gfx_microbench \
  --backend metal \
  --warmup 3 \
  --iterations 10 \
  --output .gfx-profile/gfx-microbench-macos.json \
  --calibration-output .gfx-profile/gfx-calibration-macos.json
```

### Android

```bash
adb push /path/to/build-gfx-plugin-android/output/bin/aarch64/Release/ov_gfx_microbench /data/local/tmp/openvino_gfx_android/ov_gfx_microbench
adb shell 'chmod +x /data/local/tmp/openvino_gfx_android/ov_gfx_microbench'
adb shell '
  cd /data/local/tmp/openvino_gfx_android &&
  GFX_PLUGIN_PATH=/data/local/tmp/openvino_gfx_android/libopenvino_gfx_plugin.so \
  LD_LIBRARY_PATH=/data/local/tmp/openvino_gfx_android \
  ./ov_gfx_microbench \
    --backend vulkan \
    --warmup 1 \
    --iterations 3 \
    --output /data/local/tmp/openvino_gfx_android/gfx-microbench-android.json \
    --calibration-output /data/local/tmp/openvino_gfx_android/gfx-calibration-android.json
'
```

For an OpenCL-capable Android build, use `--backend opencl` to profile the
source-artifact backend. Keep `--backend vulkan` when the goal is a legacy
SPIR-V/Vulkan comparison.

### Raspberry Pi 5

```bash
cd /path/to/gfx_eval
LD_LIBRARY_PATH=/path/to/gfx_eval/libs/Release:/path/to/gfx_eval \
./ov_gfx_microbench \
  --backend vulkan \
  --warmup 1 \
  --iterations 3 \
  --output /path/to/gfx_eval/gfx-microbench-rpi.json \
  --calibration-output /path/to/gfx_eval/gfx-calibration-rpi.json
```

If the deployed bundle includes the OpenCL backend and the target exposes a
working OpenCL GPU runtime, run the same command with `--backend opencl` to
separate OpenCL source-kernel behavior from the Vulkan/SPIR-V diagnostic path.

## Real Workload Profiling

Use detailed profiling for workload correlation:

```bash
benchmark_app \
  -m /path/to/model.xml \
  -d GFX \
  -pc \
  -niter 10
```

Recommended environment for trace export:

```bash
OV_GFX_PROFILE_TRACE=perfetto \
OV_GFX_PROFILE_TRACE_FILE=/path/to/gfx-trace.json \
benchmark_app -m <model> -d GFX -pc -niter 10
```

For full-graph accuracy runs and cross-device placement triage, prefer the
compare runner profile switch over ad hoc environment-only setup:

```bash
ov_gfx_compare_runner <model.xml> \
  --reference-device TEMPLATE \
  --dump-gfx-profile \
  --gfx-profiling-level detailed
```

The compile section includes MPSRT placement counters such as
`mpsrt_stage_backend_apple_msl_precision_fp32_count`,
`mpsrt_stage_backend_apple_mps_family_convolution_count`, and
`mpsrt_stage_family_topk_count`, plus `stage_policy_mps_*` accept/reject
counters. Runtime counters such as `mpsrt_mps_graph_gemm_request_encode_count`,
`mpsrt_mps_graph_topk_request_encode_count`, and
`mpsrt_mps_graph_sdpa_request_encode_count` identify MPSGraph-backed encodes.
Use these counters before changing Apple `f32` placement policy or assuming a
kernel route is the bottleneck.

For OpenCL source-kernel runs, the most important first checks are the selected
backend counters, `extended.target_profile`, source program creation spans, and
transfer/wait shares. A standalone OpenCL microbench result is only a kernel
experiment until the same route is represented in the plugin source-artifact
manifest and validated through GFX tests.

## Android Runbook

### AGI / Perfetto Correlation

1. Run `ov_gfx_microbench` and save the JSON files.
2. Run the real workload with `OV_GFX_PROFILE_TRACE=perfetto`.
3. Capture AGI System Profile or Frame Profile.
4. Compare:
   - `analysis.triage_hints`
   - `MB0.fixed_overhead_us`
   - `MB2.transfer_share_of_wall`
   - `MB3.wait_share_of_wall`
   - AGI GPU busy/idle gaps
   - CPU thread blocking around queue submit or fence waits

Interpretation:
- high `MB0` + idle gaps: fixed overhead and sync pressure dominate
- high `MB2.transfer_share_of_wall`: transfer-heavy path
- high `MB3.wait_share_of_wall`: sync-heavy steady-state
- low GPU busy with active CPU: driver/submit/barrier overhead is dominating

### Validation Layers

Use only for correctness, not for performance numbers.

Enable:

```bash
adb shell settings put global enable_gpu_debug_layers 1
adb shell settings put global gpu_debug_app com.intel.openvino.benchmark_app
adb shell settings put global gpu_debug_layers VK_LAYER_KHRONOS_validation
```

Run correctness workload or `ov_gfx_func_tests`, then disable:

```bash
adb shell settings delete global enable_gpu_debug_layers
adb shell settings delete global gpu_debug_app
adb shell settings delete global gpu_debug_layers
```

If the package name is different on your setup, replace `com.intel.openvino.benchmark_app` with the actual debuggable app package.

## macOS Runbook

### `xctrace`

```bash
mkdir -p .gfx-profile
xcrun xctrace record \
  --template "Time Profiler" \
  --time-limit 10s \
  --output .gfx-profile/gfx-profile.trace \
  --launch -- \
  /path/to/build-gfx-plugin-macos/output/bin/arm64/RelWithDebInfo/benchmark_app \
    -m /path/to/model.xml \
    -d GFX -pc -niter 10
```

Use this together with `OV_GFX_PROFILE_TRACE=signpost` when you want signposts in the Instruments timeline.

### Xcode Metal Workload Capture

Capture a representative inference window and compare:
- CPU timeline vs signposts
- Metal command-buffer GPU timing
- encoder setup and binding spans from the profiling report

Useful report fields:
- `extended.total_gpu_us`
- `summary.hot_segments`
- `summary.hot_nodes`
- trace-event spans for transfers, allocations, segments, counters

## Raspberry Pi 5 Runbook

### `perf stat`

```bash
perf stat -e cycles,instructions,cache-misses,branch-misses -- \
  /path/to/gfx_eval/benchmark_app \
    -m /path/to/model.xml \
    -d GFX -pc -niter 10
```

### `perf record`

```bash
perf record -g -- \
  /path/to/gfx_eval/benchmark_app \
    -m /path/to/model.xml \
    -d GFX -pc -niter 10
perf report
```

Correlate `perf` with:
- `MB0.fixed_overhead_us`
- `profile_digest.submit_count`
- `profile_digest.barrier_count`
- `profile_digest.transfer_share_of_wall`

## Calibration Artifact Workflow

Generate:

```bash
./ov_gfx_microbench --backend vulkan --warmup 1 --iterations 3 \
  --output gfx-microbench.json \
  --calibration-output gfx-calibration.json
```

Load and compare:

```bash
./ov_gfx_microbench --backend vulkan --warmup 1 --iterations 3 \
  --calibration-input gfx-calibration.json
```

The full report then exposes `loaded_calibration` with:
- `device_key_match`
- `backend_match`
- `schema_match`

To compare two artifacts:

```bash
python3 modules/gfx_plugin/tools/gfx_calibration_diff.py \
  --before old-calibration.json \
  --after new-calibration.json \
  --output calibration-diff.json
```

To smoke-test `ov_gfx_microbench` plus calibration roundtrip:

```bash
python3 modules/gfx_plugin/tools/gfx_microbench_smoke.py \
  --platform host \
  --binary /path/to/ov_gfx_microbench \
  --backend metal
```

Android variant:

```bash
python3 modules/gfx_plugin/tools/gfx_microbench_smoke.py \
  --platform android \
  --binary /path/to/ov_gfx_microbench \
  --backend vulkan \
  --plugin-path /data/local/tmp/openvino_gfx_android/libopenvino_gfx_plugin.so
```

To summarize external traces:

```bash
python3 modules/gfx_plugin/tools/gfx_external_trace_summary.py --input .gfx-profile/gfx-trace.json --kind trace_event
python3 modules/gfx_plugin/tools/gfx_external_trace_summary.py --input perf-stat.txt --kind perf_stat
python3 modules/gfx_plugin/tools/gfx_external_trace_summary.py --input perf-report.txt --kind perf_report
```

## Assumptions

- `MB1` to `MB3` are synthetic FP16 graphs.
- `adjusted_*` metrics subtract only `MB0` fixed overhead.
- `gpu_*` estimates are present only when GPU timing is available.
- Profiling-derived TFLOPS and GB/s values are for triage. Peak-hardware validation still belongs to AGI, Instruments, Xcode GPU capture, or `perf`.

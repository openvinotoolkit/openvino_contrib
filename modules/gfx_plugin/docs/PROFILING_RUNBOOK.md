# GFX Profiling Runbook

This runbook is the operational companion to `create-profiler.md`.

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

- `pipeline_creation_count`
  Meaning: pipeline or shader creation happened during infer.
  Read from: `profile_digest.pipeline_creation_count`.
  Triage meaning: prewarm or cache persistence is incomplete.

- `descriptor_update_count` or binding-preparation diagnostics
  Meaning: CPU work is spent rebinding descriptors or backend-specific binding tables.
  Read from: `profile_digest.descriptor_update_count`, `binding_prepare_in_infer`, diagnostics.
  Triage meaning: descriptor churn is visible.

## Microbench Commands

### macOS

```bash
/path/to/build-gfx-plugin-macos/output/bin/arm64/RelWithDebInfo/ov_gfx_microbench \
  --backend metal \
  --warmup 3 \
  --iterations 10 \
  --output /tmp/gfx-microbench-macos.json \
  --calibration-output /tmp/gfx-calibration-macos.json
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
OV_GFX_PROFILE_TRACE_FILE=/tmp/gfx-trace.json \
benchmark_app -m <model> -d GFX -pc -niter 10
```

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
xcrun xctrace record \
  --template "Time Profiler" \
  --time-limit 10s \
  --output /tmp/gfx-profile.trace \
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
python3 modules/gfx_plugin/tools/gfx_external_trace_summary.py --input /tmp/gfx-trace.json --kind trace_event
python3 modules/gfx_plugin/tools/gfx_external_trace_summary.py --input perf-stat.txt --kind perf_stat
python3 modules/gfx_plugin/tools/gfx_external_trace_summary.py --input perf-report.txt --kind perf_report
```

## Assumptions

- `MB1` to `MB3` are synthetic FP16 graphs.
- `adjusted_*` metrics subtract only `MB0` fixed overhead.
- `gpu_*` estimates are present only when GPU timing is available.
- Profiling-derived TFLOPS and GB/s values are for triage. Peak-hardware validation still belongs to AGI, Instruments, Xcode GPU capture, or `perf`.

<!--
Copyright (C) 2018-2026 Intel Corporation
SPDX-License-Identifier: Apache-2.0
-->

# OpenVINO YOLO Android Demo

A lightweight Android app that runs **YOLO** object detection, segmentation, pose estimation and
classification on-device with [OpenVINO](https://github.com/openvinotoolkit/openvino) through the
[OpenVINO Java API](../../java_api). It is written in **Kotlin**, uses **CameraX** for the live
camera; preprocessing, NMS and rendering are plain Kotlin + the
OpenVINO `PrePostProcessor` and Android `Canvas`.

Supported models: **YOLOv8, YOLOv10, YOLO11, YOLOv12** (anchor-free). Inference runs on **CPU**;
target ABI is **`arm64-v8a`**.

## How it works

1. **Pick a model** on the home screen — YOLO version, size (Nano … Extra-Large) and task
   (detect / segment / pose / classify). The download size is shown next to the choice.
2. **Download** — tapping *Start* streams the model's ONNX to the device over its own network, with a
   progress bar, cancel, and integrity (sha256) check. Downloaded models are cached; re-selecting one
   skips the download. *Clear cache* frees the stored models.
3. **Run** — the camera opens and inference runs live, drawing boxes / masks / keypoints / top-k
   labels on an overlay with an FPS readout.

Models are **downloaded at runtime from the model authors' own public ONNX release assets** — the app
bundles only a small manifest of those URLs plus class-name label files. The app itself ships no
model weights.

> **Model licensing.** Ultralytics YOLO weights are AGPL-3.0. This module (Apache-2.0) and the built
> APK contain **no weights**; they are fetched on-device from the upstream authors' releases. A list
> of download URLs is not the weights. *(Engineering guidance, not legal advice.)*

## Models

`app/src/main/assets/models_manifest.json` is the single source of truth for the picker (task, input
size, normalization, labels, output-format `variant`, `dynamic` flag, and each asset's URL / `bytes`
/ `sha256`). Adding or removing a model is a one-line edit — no code change.

| Version | Tasks | Source |
|---------|-------|--------|
| YOLOv8, YOLO11 | detect, segment, pose, classify | `github.com/ultralytics/assets` |
| YOLOv10 | detect | `github.com/THU-MIG/yolov10` |
| YOLOv12 | detect | `huggingface.co/jquadrino/yolo-v12-onnx` |

Class-name label files (`assets/labels/*.txt`) are bundled (names are not the weights). An optional
build flag `-PmodelBaseUrl=<host>` redirects downloads to a mirror or LAN server; the default needs
no hosting.

## Architecture

```
app/src/main/kotlin/org/intel/openvino/demo/
├── ui/            SelectActivity (picker + download), CameraActivity (live), OverlayView (Canvas), DownloadViewModel
├── data/          ModelManifest, ManifestRepository, ModelDownloader (HTTPS + progress + sha256), ModelCache, DownloadState
├── inference/     OvEngine (Core/CompiledModel/InferRequest, read_model(onnx)), ModelConfig (from the manifest)
├── camera/        CameraController (CameraX Preview + ImageAnalysis, latest-frame backpressure, own thread)
├── preprocess/    LetterboxTransform (aspect-fit + exact inverse), FramePreprocessor (YUV → u8 RGB NHWC)
└── postprocess/   Nms (class-aware, pure Kotlin), YoloDecoder + per-family decoders, Detection types
```

The camera / preprocess / inference / overlay layers are model-agnostic. Task and output-format
specifics live behind the `YoloDecoder` interface and `ModelConfig` (built from the manifest), so a
new variant is a manifest entry plus, at most, a new decoder:

- **Anchor-free** (v8 / v11 / v12): detect `[1, 4+nc, N]`; segment adds mask prototypes; pose adds
  keypoints; classify `[1, nc]`. Class-aware NMS, then the inverse letterbox to frame coordinates.
- **YOLOv10 end-to-end**: `[1, 300, 6]` (`x1,y1,x2,y2,score,class`), already NMS-free.
- **Dynamic input** (YOLOv12): the model's spatial input is chosen from the camera frame's aspect
  ratio (rounded to a multiple of 32) and the model is reshaped before compiling.

Inference runs off the UI thread on a single-threaded executor; `ImageAnalysis` keeps only the latest
frame so the preview stays smooth.

## Building and running

**Prerequisites (host):** Android SDK Platform Tools + NDK r26, SDK platform/build-tools 34, CMake
≥ 3.26, SCons ≥ 4.6, and **JDK 17 or 21** (Gradle 8.6 does not support JDK 22+).

1. **Build OpenVINO for Android (`arm64-v8a`) with the ONNX frontend.** Follow
   [openvino/docs/dev/build_android.md](https://github.com/openvinotoolkit/openvino/blob/master/docs/dev/build_android.md),
   adding `-DBUILD_java_api=ON -DOPENVINO_EXTRA_MODULES=<openvino_contrib>/modules/java_api
   -DENABLE_OV_ONNX_FRONTEND=ON`. The ONNX frontend is required to read `.onnx` on-device.
2. **Build the Java API jar:** `cd <openvino_contrib>/modules/java_api && gradle build -x test`.
3. **Place the artifacts:**
   - `openvino-<ver>-<os>.jar` → `app/libs/openvino-java-api.jar`
   - into `app/src/main/jniLibs/arm64-v8a/`: `libopenvino.so`, `libopenvino_arm_cpu_plugin.so`,
     `libopenvino_ir_frontend.so`, **`libopenvino_onnx_frontend.so`**,
     `libinference_engine_java_api.so`, `libtbb.so`, `libtbbmalloc.so`, and the NDK's
     `libc++_shared.so`.
4. **Build & install the APK:**
   ```sh
   ./gradlew :app:assembleDebug            # add -Dorg.gradle.java.home=<jdk-17-or-21> if needed
   adb install -r app/build/outputs/apk/debug/app-debug.apk
   adb shell am start -n org.intel.openvino.demo/.ui.SelectActivity
   adb logcat -s YoloDemo:I OvEngine:I     # model, input size, FPS, detection counts
   ```

## Troubleshooting

| Symptom | Cause / fix |
|---------|-------------|
| `read_model(.onnx)` fails / ONNX frontend not found | Build OpenVINO with `-DENABLE_OV_ONNX_FRONTEND=ON` and include `libopenvino_onnx_frontend.so` in `jniLibs`. |
| `UnsatisfiedLinkError` on launch | A `.so` is missing from `jniLibs/arm64-v8a/` or built for the wrong ABI. |
| No-internet / download-failed dialog | The manifest loads offline; only the model download needs network. Retry once connected. |
| Checksum mismatch | Interrupted download (the file is deleted) — tap *Start* again. |
| Boxes shifted | Inverse letterbox — verified by `LetterboxTransformTest` in `app/src/test`. |
| Low FPS | Larger sizes and segmentation are heavier on CPU; the Nano detect models are fastest. |

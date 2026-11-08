<!--
Copyright (C) 2018-2026 Intel Corporation
SPDX-License-Identifier: Apache-2.0
-->

# Face Detection Java Async Sample

This sample shows how to run face detection with OpenVINO Java API 2.0 in asynchronous mode.

## Highlights

- Uses API 2.0 classes: `Core`, `Model`, `CompiledModel`, `InferRequest`, `Tensor`, `PrePostProcessor`
- Overlaps capture and inference with multiple `InferRequest`s via `start_async()` / `wait_async()`
- Accepts a video file, camera index, or image path
- Supports `-no_show true` for headless environments

## Prerequisites

- Built OpenVINO Runtime with the Java API native library available through `INTEL_OPENVINO_DIR`
- OpenCV with Java bindings (`OpenCV_DIR`)
- Face detection model, for example `face-detection-adas-0001`

Download the model:

```bash
omz_downloader --name face-detection-adas-0001 --output_dir .
# or use any equivalent IR already converted to OpenVINO format
```

## Build

```bash
export INTEL_OPENVINO_DIR=/path/to/openvino_install
export OpenCV_DIR=/path/to/opencv
cd <openvino_contrib>/modules/java_api
gradle build -Pbuild_java_samples=true
```

## Run

```bash
export OV_JAVA_DIR=<openvino_contrib>/modules/java_api/build/libs
java -cp ".:${OV_JAVA_DIR}/openvino-*.jar:${OpenCV_DIR}/bin/*:samples/face_detection_java_async_sample/build/libs/face_detection_java_async_sample.jar" \
  Main -m face-detection-adas-0001.xml -i input.mp4 -d CPU -nireq 2 -no_show true
```

### Arguments

| Argument | Description |
|---|---|
| `-m` | Path to model `.xml` |
| `-i` | Video/image path or camera index |
| `-d` | Device name (default `CPU`) |
| `-nireq` | Number of async infer requests (default `2`) |
| `-t` | Confidence threshold (default `0.7`) |
| `-no_show` | `true` to disable GUI |

Press `Esc` / `q` in GUI mode to stop.

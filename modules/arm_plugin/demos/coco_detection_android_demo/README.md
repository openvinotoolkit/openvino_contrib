# Coco Detection Android Demo

![Running result](https://user-images.githubusercontent.com/47499836/179177513-7623b7eb-4229-4f44-b5cc-b937e93905b2.gif)

This is a demo for ARM processors Android devices. Using object detection model to reach the coco datasets' infomation. We use pre-trained models from [Open Model Zoo](https://github.com/openvinotoolkit/open_model_zoo): [ssdlite_mobilenet_v2](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/ssdlite_mobilenet_v2) for object detection in coco dataset, [efficientdet-d0-tf](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/efficientdet-d0-tf), [pelee-coco](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/pelee-coco).

The application reads frames from your device's camera or emulator's camera, and processes the network to detect the coco objects' locations and attributes. Then draw it on the image.

The current openvino engine does not currently support models in INT8 MIXed format for reasoning on ARM devices, but will support models in this format in the near future and will achieve better performance.

| Model | FP16 | FP32 |
| --- | --- | --- |
| ssdlite_mobilenet_v2 | 120ms | 120ms |
| efficientdet-d0-tf | 800ms | 800ms |
| pelee-coco | 220ms | X |

## How to run it

### Build the OpenVINO libraries for Android

Before we run the demo on ARM Android phones, we need to prepare the libraries for ARM plugin and java bingings for OpenVINO. These libraries are built from Ubuntu 18.04, and it 

- Set java environment

```bash
sudo apt-get install -y openjdk-8-jdk
export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
```

- Create work directory

```bash
mkdir openvino_android
export WORK_DIR="$(pwd)/openvino_android"
cd $WORK_DIR
```

- Clone `OpenVINO` and `OpenVINO Contrib` repositories(Use 2021.4.1 branch).

```bash
git clone --recurse-submodules --shallow-submodules --depth 1 --branch=2021.4.1 https://github.com/openvinotoolkit/openvino.git "$WORK_DIR/openvino"
git clone --recurse-submodules --shallow-submodules --depth 1 --branch=2021.4 https://github.com/openvinotoolkit/openvino_contrib.git "$WORK_DIR/openvino_contrib"
```

- Download Android NDK and set environment for it

```bash
wget https://dl.google.com/android/repository/android-ndk-r20-linux-x86_64.zip
unzip android-ndk-r20-linux-x86_64.zip
```

- Build OpenVINO and ARM plugin for ARM64

```bash
mkdir "$WORK_DIR/openvino/build"
cmake -DCMAKE_BUILD_TYPE=Release \
      -DTHREADING=SEQ \
      -DIE_EXTRA_MODULES="$WORK_DIR/openvino_contrib/modules" \
      -DBUILD_java_api=ON \
      -DCMAKE_TOOLCHAIN_FILE="$WORK_DIR/android-ndk-r20/build/cmake/android.toolchain.cmake" \
      -DANDROID_ABI=arm64-v8a \
      -DANDROID_PLATFORM=29 \
      -DANDROID_STL=c++_shared \
      -DENABLE_SAMPLES=OFF \
      -DENABLE_OPENCV=OFF \
      -DENABLE_CLDNN=OFF \
      -DENABLE_VPU=OFF \
      -DENABLE_GNA=OFF \
      -DENABLE_MYRIAD=OFF \
      -DENABLE_TESTS=OFF \
      -DENABLE_GAPI_TESTS=OFF \
      -DENABLE_BEH_TESTS=OFF \
      -B "$WORK_DIR/openvino/build" \
      -S "$WORK_DIR/openvino"
make -C "$WORK_DIR/openvino/build" --jobs=$(nproc --all)
```

- OpenVINO binaries could be stripped to reduce their size:

```bash
$WORK_DIR/android-ndk-r20/toolchains/aarch64-linux-android-4.9/prebuilt/linux-x86_64/aarch64-linux-android/bin/strip $WORK_DIR/openvino/bin/aarch64/Release/lib/*.so
```

The built results are in `$WORK_DIR/openvino/bin/aarch64/Release/lib`. We will use them later.

> Please confirm that your `plugins.xml` in `$WORK_DIR/openvino/bin/aarch64/Release/lib` contains plugin name `"CPU"` and `libarmPlugin.so` library file in `$WORK_DIR/openvino/bin/aarch64/Release/lib`

### Import demo project on Android Studio

In this step, we will import demo project to infer object detection.

- Choose and download [Android Studio](https://developer.android.com/studio) on your PC.

- Clone latest branch of OpenVINO Contrib.

```bash
git clone https://github.com/openvinotoolkit/openvino_contrib.git "$WORK_DIR/demo"
```

- Select "File -> Open", and import demo project in `"$WORK_DIR/demo/openvino_contrib/modules/java_api/demos/coco_detection_android_demo"`.

- Copy libraries and model files to the corresponding folder.

  1. Clone `"$WORK_DIR/openvino/bin/aarch64/Release/lib/inference_engine_java_api.jar"` to `app/libs` folder.
  2. Clone `"$WORK_DIR/openvino/bin/aarch64/Release/lib/*.so"` and `"$WORK_DIR/android-ndk-r20/sources/cxx-stl/llvm-libc++/libs/arm64-v8a/libc++_shared.so"` to `"app/src/main/jniLibs/arm64-v8a"`
  3. Clone `"$WORK_DIR/openvino/bin/aarch64/Release/lib/plugins.xml"` to `"app/src/main/assets"`
  4. Download and convert model "ssdlite_mobilenet_v2" [or pelee-coco, efficientdet-d0-tf] with Open Model Zoo in following steps and copy `"$WORK_DIR/open_model_zoo/tools/downloader/intel/ssdlite_mobilenet_v2/FP32/ssdlite_mobilenet_v2.xml"`, `"$WORK_DIR/open_model_zoo/tools/downloader/intel/ssdlite_mobilenet_v2/FP32/ssdlite_mobilenet_v2.bin"` to `"app/src/main/assets"`

```bash
git clone --depth 1 https://github.com/openvinotoolkit/open_model_zoo "$WORK_DIR/open_model_zoo"
cd "$WORK_DIR/open_model_zoo/tools/downloader"
python3 -m pip install -r requirements.in
omz_downloader  --name ssdlite_mobilenet_v2 --output_dir $WORK_DIR/open_model_zoo/tools/downloader
omz_converter --name ssdlite_mobilenet_v2 --download_dir $WORK_DIR/open_model_zoo/tools/downloader --precision FP32
```

- Add OpenCV dependency to project

  1. Download [OpenCV SDK for Android](https://github.com/opencv/opencv/releases/download/4.5.0/opencv-4.5.0-android-sdk.zip) and unpack it.
  2. Import OpenCV module: select "File -> New -> ImportModule", and sepcify a path to unpacked SDK and set module name to "ocv".
  3. Replace `compileSdkVersion 26`, `targetSdkVersion 26` to `compileSdkVersion 32`, `targetSdkVersion 32` in `"$WORK_DIR/coco_detection_android_demo/ocv/build.gradle"`

- Start a ARM-based Android Emulator.

  1. Using `AVD Manager -> Create Virtual Device`, and choose one virtual device.
  2. Select a system image with `arm64-v8a`.

- Run it!

> The first time when you run the demo application on your device, your need to grant camera permission. Then run it again.
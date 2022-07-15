# Vehicle Detection and Recognition Android Demo

![Running result](https://user-images.githubusercontent.com/47499836/179177513-7623b7eb-4229-4f44-b5cc-b937e93905b2.gif)

This is a demo for ARM processors Android devices. Using vehicle detection and recognition model to reach the vehicles's infomation. We use pre-trained models from [Open Model Zoo](https://github.com/openvinotoolkit/open_model_zoo): [vehicle-detection-0200](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/intel/vehicle-detection-0200) for object detection and [vehicle-attributes-recognition-barrier-0039](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/intel/vehicle-attributes-recognition-barrier-0039) for image classification.

The application reads frames from your device's camera or a video file, and processes the network to detect the vehicle's locations and attributes. Then draw it on the image.

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

- Clone `OpenVINO` and `OpenVINO` Contrib repositories(Use 2021.4.1 branch).

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

In this step, we will import demo project to infer vehicle detection and recognition models.

- Choose and download [Android Studio](https://developer.android.com/studio) on your PC.

- Clone latest branch of OpenVINO Contrib.

```bash
git clone https://github.com/openvinotoolkit/openvino_contrib.git "$WORK_DIR/demo"
```

- Select "File -> Open", and import demo project in `"$WORK_DIR/demo/openvino_contrib/modules/java_api/samples/vehicle_detection_and_recognition_android_demo"`.

- Copy libraries and model files to the corresponding folder.

1. Clone `"$WORK_DIR/openvino/bin/aarch64/Release/lib/inference_engine_java_api.jar"` to `app/libs` folder.

2. Clone `"$WORK_DIR/openvino/bin/aarch64/Release/lib/*.so"` and `"$WORK_DIR/android-ndk-r20/sources/cxx-stl/llvm-libc++/libs/arm64-v8a/libc++_shared.so"` to `"app/src/main/jniLibs/arm64-v8a"`
3. Clone `"$WORK_DIR/openvino/bin/aarch64/Release/lib/plugins.xml"` to `"app/src/main/assets"`
4. Download model "vehicle-attributes-recognition-barrier-0039" and "vehicle-detection-0200" with Open Model Zoo in following steps and copy `"$WORK_DIR/open_model_zoo/tools/downloader/intel/vehicle-attributes-recognition-barrier-0039/FP32/vehicle-attributes-recognition-barrier-0039.xml"`, `"$WORK_DIR/open_model_zoo/tools/downloader/intel/vehicle-attributes-recognition-barrier-0039/FP32/vehicle-attributes-recognition-barrier-0039.bin"`, `"$WORK_DIR/open_model_zoo/tools/downloader/intel/vehicle-detection-0200/FP32/vehicle-detection-0200.xml"` , `"$WORK_DIR/open_model_zoo/tools/downloader/intel/vehicle-detection-0200/FP32/vehicle-detection-0200.bin"` to `"app/src/main/assets"`
5. Download a test car image and rename to `"cars.png"`, then copy it to `"app/src/main/assets"` (Using video and camera is in progress)

```bash
git clone --depth 1 https://github.com/openvinotoolkit/open_model_zoo "$WORK_DIR/open_model_zoo"
cd "$WORK_DIR/open_model_zoo/tools/downloader"
python3 -m pip install -r requirements.in
python3 downloader.py --name vehicle-attributes-recognition-barrier-0039  --precision FP32
python3 downloader.py --name vehicle-detection-0200 --precision FP32
```

- Add OpenCV dependency to project

1. Download [OpenCV SDK for Android](https://github.com/opencv/opencv/releases/download/4.5.0/opencv-4.5.0-android-sdk.zip) and unpack it.
2. Import OpenCV module: select "File -> New -> ImportModule", and sepcify a path to unpacked SDK and set module name to "ocv".
3. Replace `compileSdkVersion 26`, `targetSdkVersion 26` to `compileSdkVersion 32`, `targetSdkVersion 32` in `"$WORK_DIR/vehicle_detection_and_recognition_android_demo/ocv/build.gradle"`

- Start a ARM-based Android Emulator.

1. Using `AVD Manager -> Create Virtual Device`, and choose one virtual device.
2. Select a system image with `arm64-v8a`.

- Run it!

> The first time when you run the demo application on your device, your need to grant camera permission. Then run it again.
# Coco Detection Android Demo

![Running result](https://user-images.githubusercontent.com/47499836/189129594-2634e176-5a5b-4051-b713-ae9574a8c3da.png)

This is a demo for ARM processors Android devices. Using object detection model to reach the coco datasets' infomation. We use pre-trained models from [Open Model Zoo](https://github.com/openvinotoolkit/open_model_zoo): [ssdlite_mobilenet_v2](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/ssdlite_mobilenet_v2) for object detection in coco dataset, [efficientdet-d0-tf](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/efficientdet-d0-tf), [pelee-coco](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/pelee-coco).

The application reads frames from your device's camera or emulator's camera, and processes the network to detect the coco objects' locations and attributes. Then draw it on the image.

The current openvino engine does not currently support models in INT8 MIXed format for reasoning on ARM devices, but will support models in this format in the near future and will achieve better performance.

Otherwise, there is no difference between FP16 and FP32 for CPU, the plugin will convert it automatically to FP32. 

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

- Clone `OpenVINO` and `OpenVINO Contrib` repositories (Use master branch for Java API 2.0).

```bash
git clone --recurse https://github.com/openvinotoolkit/openvino.git "$WORK_DIR/openvino"
git clone --recurse https://github.com/openvinotoolkit/openvino_contrib.git "$WORK_DIR/openvino_contrib"
```

- Download Android NDK and set environment for it. (If you need proxy, you need to set specific url to XXX, or just remove `--no_https --proxy=http --proxy_host=XXX --proxy_port=XXX`)

```bash
mkdir "$WORK_DIR/android-tools"
wget https://dl.google.com/android/repository/commandlinetools-linux-7583922_latest.zip
unzip commandlinetools-linux-7583922_latest.zip
yes | ./cmdline-tools/bin/sdkmanager --sdk_root="$WORK_DIR/android-tools" --licenses --no_https --proxy=http --proxy_host=XXX --proxy_port=XXX
./cmdline-tools/bin/sdkmanager --sdk_root="$WORK_DIR/android-tools" --install "ndk-bundle" --no_https --proxy=http --proxy_host=XXX --proxy_port=XXX
```

- Build OpenVINO and ARM plugin for ARM64

```bash
mkdir "$WORK_DIR/openvino_build" "$WORK_DIR/openvino_install"
cmake -GNinja \
      -DVERBOSE_BUILD=ON \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_TOOLCHAIN_FILE="$WORK_DIR/android-tools/ndk-bundle/build/cmake/android.toolchain.cmake" \
      -DANDROID_ABI=arm64-v8a \
      -DANDROID_STL=c++_shared \
      -DANDROID_PLATFORM=29 \
      -DENABLE_SAMPLES=ON \
      -DENABLE_INTEL_MYRIAD=OFF \
      -DENABLE_INTEL_MYRIAD_COMMON=OFF \
      -DBUILD_java_api=ON \
      -DTHREADING=SEQ \
      -DIE_EXTRA_MODULES="$WORK_DIR/openvino_contrib/modules" \
      -DARM_COMPUTE_SCONS_JOBS=$(nproc) \
      -DCMAKE_INSTALL_PREFIX="$WORK_DIR/openvino_install" \
      -B "$WORK_DIR/openvino_build" -S "$WORK_DIR/openvino"
ninja -C "$WORK_DIR/openvino_build"
ninja -C "$WORK_DIR/openvino_build" install
```

The built results are in `$WORK_DIR/openvino_install/runtime/lib/aarch64`. We will use them later.

> Please confirm that your `plugins.xml` in `$WORK_DIR/openvino_install/runtime/lib/aarch64` contains plugin name `"CPU"`.

- Download and convert model "ssdlite_mobilenet_v2" [or pelee-coco, efficientdet-d0-tf] with Open Model Zoo

```bash
git clone --depth 1 https://github.com/openvinotoolkit/open_model_zoo "$WORK_DIR/open_model_zoo"
cd "$WORK_DIR/open_model_zoo/tools/downloader"
python3 -m pip install -r requirements.in
omz_downloader  --name ssdlite_mobilenet_v2 --output_dir $WORK_DIR/open_model_zoo/tools/downloader
omz_converter --name ssdlite_mobilenet_v2 --download_dir $WORK_DIR/open_model_zoo/tools/downloader --precision FP16
```

### Import demo project on Android Studio

In this step, we will import demo project to infer object detection.

- Choose and download [Android Studio](https://developer.android.com/studio) on your PC.

- Clone latest branch of OpenVINO Contrib.

```bash
git clone https://github.com/openvinotoolkit/openvino_contrib.git "$WORK_DIR/demo"
```

- Select "File -> Open", and import demo project in `"$WORK_DIR/demo/openvino_contrib/modules/java_api/demos/coco_detection_android_demo"`.

- Copy libraries and model files to the corresponding folder.

  1. Clone `"$WORK_DIR/openvino_contrib/modules/java_api/build/libs/java_api.jar"` to `app/libs` folder, and add it as library.
  2. Clone `"$WORK_DIR/openvino_install/runtime/lib/aarch64/*.so"` and `"$WORK_DIR/android-tools/ndk-bundle/sources/cxx-stl/llvm-libc++/libs/arm64-v8a/libc++_shared.so"` to `"app/src/main/jniLibs/arm64-v8a"`
  3. Clone `"$WORK_DIR/openvino_install/runtime/lib/aarch64/plugins.xml"` to `"app/src/main/assets"`
  4. Copy `"$WORK_DIR/open_model_zoo/tools/downloader/intel/ssdlite_mobilenet_v2/FP16/ssdlite_mobilenet_v2.xml"`, `"$WORK_DIR/open_model_zoo/tools/downloader/intel/ssdlite_mobilenet_v2/FP16/ssdlite_mobilenet_v2.bin"` to `"app/src/main/assets"`

- Add OpenCV dependency to project

  1. Download [OpenCV SDK for Android](https://github.com/opencv/opencv/releases/download/4.5.0/opencv-4.5.0-android-sdk.zip) and unpack it.
  2. Import OpenCV module: select "File -> New -> ImportModule", and sepcify a path to unpacked SDK and set module name to "ocv".
  3. Replace `compileSdkVersion 26`, `targetSdkVersion 26` to `compileSdkVersion 32`, `targetSdkVersion 32` in `"$WORK_DIR/coco_detection_android_demo/ocv/build.gradle"`

- Start a ARM-based Android Emulator.

  1. Using `AVD Manager -> Create Virtual Device`, and choose one virtual device.
  2. Select a system image with `arm64-v8a`.

- Run it!

> The first time when you run the demo application on your device, your need to grant camera permission. Then run it again.
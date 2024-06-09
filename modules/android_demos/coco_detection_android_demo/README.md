# Coco Detection Android Demo

![Running result](https://user-images.githubusercontent.com/47499836/189129594-2634e176-5a5b-4051-b713-ae9574a8c3da.png)

This is a demo for Android ARM devices. Using object detection model to reach the coco datasets' information. 

## How to run it

### Build the OpenVINO libraries for Android

To build the OpenVINO library for an Android system, please follow these step-by-step [instruction](https://github.com/openvinotoolkit/openvino/blob/master/docs/dev/build_android.md) in full. 
After successful completion, you can move on to the next step.

### Re-build the OpenVINO libraries for Java API
_Please save the state of the environment variables_ 

For more information, please refer to [these instructions](../../java_api/README.md)
  ```sh
  # Clone OpenVINO™ contrib repository 
  git clone --recursive https://github.com/openvinotoolkit/openvino_contrib $OPV_HOME_DIR/openvino_contrib
  # Re-configure, created in the previous step, the OpenVINO™ CMake project for Java API
  cmake -S $OPV_HOME_DIR/openvino \
        -B $OPV_HOME_DIR/openvino-build \
        -DCMAKE_INSTALL_PREFIX=$OPV_HOME_DIR/openvino-install \
        -DBUILD_java_api=ON \
        -DOPENVINO_EXTRA_MODULES=$OPV_HOME_DIR/openvino_contrib/modules/java_api
  # Re-build OpenVINO™ project 
  cmake --build $OPV_HOME_DIR/openvino-build --parallel
  # Re-install OpenVINO™ project 
  cmake --install $OPV_HOME_DIR/openvino-build
  ```

### Build the OpenVINO JAVA library for Android
For more information, please refer to [these instructions](../../java_api/README.md)
  ```sh
  gradle build --project-dir $OPV_HOME_DIR/openvino_contrib/modules/java_api
  ```

### Preparing a demo to run it
  ```sh
  export ANDROID_DEMO_PATH=$OPV_HOME_DIR/openvino_contrib/modules/android_demos/coco_detection_android_demo
  # export ANDROID_DEMO_PATH=~/CLionProjects/openvino_contrib/modules/android_demos/coco_detection_android_demo
  mkdir -p $ANDROID_DEMO_PATH/app/libs
  cp $OPV_HOME_DIR/openvino_contrib/modules/java_api/build/libs/* $ANDROID_DEMO_PATH/app/libs/
  
  mkdir -p $ANDROID_DEMO_PATH/app/src/main/jniLibs/arm64-v8a
  cp -r $OPV_HOME_DIR/openvino-install/runtime/lib/aarch64/* $ANDROID_DEMO_PATH/app/src/main/jniLibs/arm64-v8a/
  cp -r $OPV_HOME_DIR/one-tbb-install/lib/* $ANDROID_DEMO_PATH/app/src/main/jniLibs/arm64-v8a/
  cp -r $ANDROID_NDK_PATH/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so $ANDROID_DEMO_PATH/app/src/main/jniLibs/arm64-v8a/

  mkdir -p $ANDROID_DEMO_PATH/app/src/main/assets/
  cp -r $OPV_HOME_DIR/mobelinet-v3-tf/* $ANDROID_DEMO_PATH/app/src/main/assets/
  
  wget https://github.com/opencv/opencv/releases/download/4.5.0/opencv-4.5.0-android-sdk.zip --directory-prefix $OPV_HOME_DIR
  unzip $OPV_HOME_DIR/opencv-4.5.0-android-sdk.zip -d $OPV_HOME_DIR
  export ANDROID_OCV_SDK_PATH=$OPV_HOME_DIR/OpenCV-android-sdk
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
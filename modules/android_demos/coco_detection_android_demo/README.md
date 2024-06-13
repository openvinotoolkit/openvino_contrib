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
  mkdir -p $ANDROID_DEMO_PATH/app/libs
  cp $OPV_HOME_DIR/openvino_contrib/modules/java_api/build/libs/* $ANDROID_DEMO_PATH/app/libs/
  
  mkdir -p $ANDROID_DEMO_PATH/app/src/main/jniLibs/arm64-v8a
  cp -r $OPV_HOME_DIR/openvino-install/runtime/lib/aarch64/* $ANDROID_DEMO_PATH/app/src/main/jniLibs/arm64-v8a/
  cp -r $OPV_HOME_DIR/one-tbb-install/lib/* $ANDROID_DEMO_PATH/app/src/main/jniLibs/arm64-v8a/
  cp -r $ANDROID_NDK_PATH/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so $ANDROID_DEMO_PATH/app/src/main/jniLibs/arm64-v8a/

  cp -r $OPV_HOME_DIR/mobelinet-v3-tf/* $ANDROID_DEMO_PATH/app/src/main/assets/
  
  wget https://github.com/opencv/opencv/releases/download/4.10.0/opencv-4.10.0-android-sdk.zip --directory-prefix $OPV_HOME_DIR
  unzip $OPV_HOME_DIR/opencv-4.10.0-android-sdk.zip -d $OPV_HOME_DIR
  mv $OPV_HOME_DIR/OpenCV-android-sdk/sdk $ANDROID_DEMO_PATH/OpenCV
  ```

Please rename jar library that project works correct , e.g.
  ```sh
  # Release version can be changed
  mv $ANDROID_DEMO_PATH/app/libs/openvino-2024.2-linux-x86_64.jar $ANDROID_DEMO_PATH/app/libs/openvino-java-api.jar
  ```

### Import demo project on Android Studio

- Choose and download [Android Studio](https://developer.android.com/studio) on your PC.

- Select "File -> Open", and import demo project in `$OPV_HOME_DIR/openvino_contrib/modules/android_demos/coco_detection_android_demo`.

- Integrate the OpenCV library from `$OPV_HOME_DIR/OpenCV-android-sdk` into the project according to the following [instructions](https://docs.opencv.org/4.x/d5/df8/tutorial_dev_with_OCV_on_Android.html). _Please do not change the name of the SDK to be integrated_

- Build and run demo

> The first time when you run the demo application on your device, your need to grant camera permission. Then run it again.

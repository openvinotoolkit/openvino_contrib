# OpenVINO™ ARM-plugin Face Recognition Demo

This demo recognizes faces on camera frames using OpenVINO™ ARM-plugin on Android smartphones*.

## 1. Build OpenVINO™ for Android on ARM

Build OpenVINO™ with Java bindings and ARM plugin for ARM Android platform using the following [instruction](https://opencv.org/face-recognition-on-android-using-openvino-toolkit-with-arm-plugin/).

## 2. Install requirements  

Install gradle build tool to build the project using console:
```
sudo apt update
sudo apt install snapd
sudo snap install gradle --classic
```

*Demo could be built on Linux-based systems.*

## 3. Setup environment
Define the following environment variables:
```
export OPENVINO_ARM_LIBRARY_PATH=<path to openvino arm-plugin *.so libraries> // e.g. ~/android_ov/openvino/bin/aarch64/Release/lib/
export ANDROID_NDK_LIBRARY_PATH=<path to android-ndk *.so libraries> // e.g. ~/android_ov/android-ndk-r20/
export ARM_PROCESSOR_CONFIGURATION=<please set arm64-v8a or armeabi-v7a>
export OPENCV_SDK_PATH=<path to opencv android sdk> // e.g. ~/android_ov/opencv-4.5.0-android-sdk/OpenCV-android-sdk
export OMZ_NETWORKS_PATH=<path to models (.xml + .bin) from Open Model ZOO for use in demo> // e.g. ~/android_ov/open_model_zoo/tools/downloader/intel/face-detection-adas-0001/FP32/
```

## 4. Build the demo
```
gradle assembleDebug
```

## 5. Send the demo apk to ARM Android smartphone
```
gradle installDebug
```

**Steps 3, 4, 5 can also be done in Android Studio**

## 6. Finally, run apk on smartphone

*Tested on a Samsung Galaxy S21 smartphone.
# OpenVINO™ ARM-plugin Face Recognition Demo

This demo recognizes faces on camera frames using OpenVINO™ ARM-plugin on Android smartphones*.

## 1. Build OpenVINO™ ARM-plugin for Android

The build OpenVINO™ ARM-plugin for Android instruction can be find in this [link](https://opencv.org/face-recognition-on-android-using-openvino-toolkit-with-arm-plugin/).

## 2. Install requirements  

Please install gradle for build project on console:
```
sudo apt update
sudo apt install snapd
sudo snap install gradle --classic
```

*Demo could build on linux based systems.*

## 3. Setup environment
Please setup next environment variables on console:
```
export OPENVINO_ARM_LIBRARY_PATH=<path to openvino arm-plugin *.so libraries> // e.g. ~/android_ov/openvino/bin/aarch64/Release/lib/
export ANDROID_NDK_LIBRARY_PATH=<path to android-ndk *.so libraries> // e.g. ~/android_ov/android-ndk-r20/
export ARM_PROCESSOR_CONFIGURATION=<please set arm64-v8a or armeabi-v7a>
export OPENCV_SDK_PATH=<path to opencv android sdk> // e.g. ~/android_ov/opencv-4.5.0-android-sdk/OpenCV-android-sdk
export OMZ_NETWORKS_PATH=<path to models (.xml + .bin) from Open Model ZOO for use in demo> // e.g. ~/android_ov/open_model_zoo/tools/downloader/intel/face-detection-adas-0001/FP32/
```

## 4. Build demo
```
gradle assembleDebug
```

## 5. Send demo apk on smartphone
```
gradle installDebug
```

**Steps 3, 4, 5 can also be done in Android Studio**

## 6. Finally, run apk on smartphone

*Tested on a Samsung Galaxy S21 smartphone.
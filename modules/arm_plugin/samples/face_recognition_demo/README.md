# OpenVINO™ ARM-plugin Face Recognition Demo

This demo shows Face Recognition task OpenVINO™ ARM-plugin working on android smartphones*.

## Build OpenVINO™ ARM-plugin for Android

The build OpenVINO™ ARM-plugin for Android instruction can be find in this [link](https://opencv.org/face-recognition-on-android-using-openvino-toolkit-with-arm-plugin/).

## Install requirements  

Please install gradle for build project on console:
```
sudo apt update
sudo apt install snapd
sudo snap install gradle --classic
```

*Demo could build on linux based systems.*

## Setup environment
Please setup next environment variables on console:
```
export OPENVINO_ARM_LIBRARY_PATH=<path to openvino arm-plugin *.so libraries> // e.g. ~/android_ov/openvino/bin/aarch64/Release/lib/
export ANDROID_NDK_LIBRARY_PATH=<path to android-ndk *.so libraries> // e.g. ~/android_ov/android-ndk-r20/
export ARM_PROCESSOR_CONFIGURATION=<please set arm64-v8a or armeabi-v7a>
```

*Tested on a Samsung Galaxy S21 smartphone.
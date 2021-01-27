
# [OpenVINO™ Toolkit](https://01.org/openvinotoolkit) - ARM CPU plugin

OpenVINO™ ARM CPU plugin is developed in order to enable deep neural networks inference on ARM CPUs, using OpenVINO™ API. The plugin uses [ARM Compute Library\*] as a backend.

## Supported Platforms
OpenVINO™ ARM CPU plugin is supported and validated on the following platforms: 

Host  | OS
------------- | -------------
Raspberry Pi 4 Model B   | Debian 10.3

## Distribution
OpenVINO™ ARM CPU plugin is not included into Intel® Distribution of OpenVIVO™. To use the plugin, it should be built from source code.

## How to build
### Approach #1: build OpenVINO and the plugin simultaneously
You may build OpenVINO™ with the plugin simultaneously following the guideline with some changes:

1. Clone `openvino_contrib` repository after you cloned `openvino` repository:
```
git clone --recurse-submodules --single-branch --branch=master https://github.com/openvinotoolkit/openvino_contrib.git 
```
2. Run Docker container with mounted both openvino and openvino_contrib source code. If you're using native compilation just skip this step:
```
docker run -it -v /absolute/path/to/openvino:/openvino -v /absolute/path/to/openvino_contrib:/openvino_contrib ie_cross_armhf /bin/bash 
```
3. Install scons in the container if you're using cross-compilation. If you do native compilation, install scons on build machine:
```
apt-get install scons
```
4. Run cmake with [extra modules flags]:
```
 cmake -DCMAKE_BUILD_TYPE=Release \
       -DCMAKE_TOOLCHAIN_FILE="../cmake/arm.toolchain.cmake" \
       -DTHREADS_PTHREAD_ARG="-pthread" \
       -DIE_EXTRA_MODULES=/openvino_contrib/modules \
       -DBUILD_java_api=OFF \
       -DBUILD_mo_pytorch=OFF ..
```

As soon as make command is finished you can find the resulting OpenVINO™ binaries in the `openvino/bin/armv7l` and the plugin `libarmPlugin.so` in `openvino/bin/armv7l/Release/lib`.

### Approach #2: build OpenVINO and the plugin consequentially
In order to build the plugin, you must prebuild OpenVINO package from sources using [this guideline](https://github.com/openvinotoolkit/openvino/wiki/BuildingCode#building-for-different-oses).

Afterwards plugin build procedure is as following:

1.  Install necessary build dependencies:
```
sudo apt-get update  
sudo apt-get install -y git cmake  scons build-essentials
```
2.  Go to cloned plugin repository:
 ```
cd arm-plugin
```
3.  Retrieve ArmCompute  submodule (if not yet downloaded):
 ```
git submodule update --init --recursive
```
4.  Prepare a build folder:
```
mkdir build && cd build
```
5.  Build plugin:
```
cmake -DInferenceEngineDeveloperPackage_DIR=<path to OpenVINO package build folder> \  
-DCMAKE_BUILD_TYPE=Release .. && make
```
### Approach #3: build OpenVINO and the plugin automatically using Docker
Also OpenVINO™ and ARM CPU plugin could be built in Docker containers for [32-bit](Dockerfile.RPi32) and [64-bit](Dockerfile.RPi64) Debian.

## Supported Configuration Parameters
The plugin supports the configuration parameters listed below. All parameters must be set before calling `InferenceEngine::Core::LoadNetwork()` in order to take effect. When specifying key values as raw strings (that is, when using Python API), omit the `KEY_` prefix.

Parameter name  | Parameter values  | Default  | Description
------------- | ------------- | ------------- | -------------
`KEY_CPU_THROUGHPUT_STREAMS`   | `KEY_CPU_THROUGHPUT_NUMA`, `KEY_CPU_THROUGHPUT_AUTO`, or positive integer values  | 1  | Specifies number of CPU "execution" streams for the throughput mode. Upper bound for the number of inference requests that can be executed simultaneously. All available CPU cores are evenly distributed between the streams.
`KEY_CPU_BIND_THREAD`   | YES/NUMA/NO  | YES  | Binds inference threads to CPU cores.

## Supported Layers and Limitations
The plugin supports IRv10 and higher. The list of supported layers and its limitations are defined in [arm_opset.md](src/arm_converter/arm_opset.md).

## Supported Model Formats
* FP32 – Supported and preferred
* FP16 – Supported
* I8 – Not supported

## Supported Input Precision
* FP32 - Supported
* FP16 - Supported
* U8 - Supported
* U16 - Supported
* I8 - Not supported
* I16 - Not supported

## Supported Output Precision 
* FP32 – Supported
* FP16 - Supported

## Supported Input Layout
* NCDHW – Not supported
* NCHW - Supported
* NHWC - Supported
* NC - Supported

## Sample
You could verify the plugin by running [OpenVINO™ samples]. You can find C++ samples in `openvino/bin/armv7l/Release`. Let's try to run Object Detection for SSD sample:
1. Download and convert vehicle-license-plate-detection-barrier-0123 model using [model downloader] and [model converter] tools from Open Model Zoo.
2. Download a vehicle image, for instance, [this image].
3. Run object detection sample:
```
./object_detection_sample_ssd -m vehicle-license-plate-detection-barrier-0123.xml -i car_1.bmp -d ARM
```

On the output image you should see 2 cars enclosed in purple rectangles and a front plate enclosed in grey rectangle:

![](https://user-images.githubusercontent.com/1412335/103134082-83458000-46bf-11eb-90b5-ef3b7ccd23ff.jpg)

One could try the plugin suitability and performance using not only OpenVINO™ samples but also Open Model Zoo demo applications and corresponding models. Demo applications could be built in accordance with [this guideline].

The plugin utilizes standard OpenVINO™ plugin infrastructure so could be tried with any demo based on supported DL model. In order to run the chosen demo with the plugin one should use "–d ARM" command-line parameter.

## License
OpenVINO™ ARM CPU plugin is licensed under [Apache License Version 2.0](LICENSE).
By contributing to the project, you agree to the license and copyright terms therein
and release your contribution under these terms.

## How to Contribute
We welcome community contributions to `openvino_contrib` repository. 
If you have an idea how to improve the modules, please share it with us. 
All guidelines for contributing to the repository can be found [here](../../CONTRIBUTING.md).

---
\* Other names and brands may be claimed as the property of others.

[ARM Compute Library\*]:https://github.com/ARM-software/ComputeLibrary
[extra modules flags]:https://github.com/openvinotoolkit/openvino_contrib#how-to-build-openvino-with-extra-modules
[OpenVINO™ samples]:https://docs.openvinotoolkit.org/latest/openvino_docs_IE_DG_Samples_Overview.html
[model downloader]:https://github.com/openvinotoolkit/open_model_zoo/blob/master/tools/downloader/README.md#model-downloader-usage
[model converter]:https://github.com/openvinotoolkit/open_model_zoo/blob/master/tools/downloader/README.md#model-converter-usage
[this image]:https://github.com/openvinotoolkit/openvino/blob/master/scripts/demo/car_1.bmp
[this guideline]:https://github.com/openvinotoolkit/open_model_zoo/blob/master/demos/README.md#build-the-demo-applications

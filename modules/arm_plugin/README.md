
# [OpenVINO™ Toolkit](https://01.org/openvinotoolkit) - ARM CPU plugin

OpenVINO™ ARM CPU plugin is developed in order to enable deep neural networks inference on ARM CPUs, using OpenVINO™ API. The plugin uses [ARM Compute Library\*] as a backend.

## Supported Platforms
OpenVINO™ ARM CPU plugin is supported and validated on the following platforms: 

Host  | OS
------------- | -------------
Raspberry Pi* 4 Model B   | Debian* 9 (32-bit)
Raspberry Pi* 4 Model B   | Debian* 10.3 (32-bit)
Raspberry Pi* 4 Model B   | Ubuntu* 18.04 (64-bit)
Apple* Mac mini with M1   | MacOS* 11.1 (64-bit)

## Distribution
OpenVINO™ ARM CPU plugin is not included into Intel® Distribution of OpenVINO™. To use the plugin, it should be built from source code.

## Get Started

1. [Build ARM CPU plugin](https://github.com/openvinotoolkit/openvino_contrib/wiki/How-to-build-ARM-CPU-plugin)
2. [Prepare models](https://github.com/openvinotoolkit/openvino_contrib/wiki/How-to-prepare-models)
3. [Run IE samples](https://github.com/openvinotoolkit/openvino_contrib/wiki/How-to-run-IE-samples)

## Supported Configuration Parameters
The plugin supports the configuration parameters listed below. All parameters must be set before calling `InferenceEngine::Core::LoadNetwork()` in order to take effect. When specifying key values as raw strings (that is, when using Python API), omit the `KEY_` prefix.

Parameter name  | Parameter values  | Default  | Description
------------- | ------------- | ------------- | -------------
`KEY_CPU_THROUGHPUT_STREAMS`   | `KEY_CPU_THROUGHPUT_NUMA`, `KEY_CPU_THROUGHPUT_AUTO`, or non negative integer values  | 1  | Specifies number of CPU "execution" streams for the throughput mode. Upper bound for the number of inference requests that can be executed simultaneously. All available CPU cores are evenly distributed between the streams.
`KEY_CPU_BIND_THREAD`   | YES/NUMA/NO  | YES  | Binds inference threads to CPU cores. Enabled only if OpenVINO™ is built with TBB that supports affinity configuration
`KEY_CPU_THREADS_NUM` | positiv integer values| Limit `#threads` that are used by Inference Engine for inference on the CPU

## Supported Layers and Limitations
The plugin supports IRv10 and higher. The list of supported layers and its limitations are defined in [arm_opset.md](docs/arm_opset.md).

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
[build it from source]:https://docs.opencv.org/master/d7/d9f/tutorial_linux_install.html
[Object Detection for SSD sample]:https://docs.openvinotoolkit.org/latest/openvino_inference_engine_samples_object_detection_sample_ssd_README.html
[Model Optimizer]:https://github.com/openvinotoolkit/openvino/tree/master/model-optimizer
[model downloader]:https://github.com/openvinotoolkit/open_model_zoo/blob/master/tools/downloader/README.md#model-downloader-usage
[model converter]:https://github.com/openvinotoolkit/open_model_zoo/blob/master/tools/downloader/README.md#model-converter-usage
[this image]:https://github.com/openvinotoolkit/openvino/blob/master/scripts/demo/car_1.bmp
[Intermediate Representation]:https://docs.openvinotoolkit.org/latest/openvino_docs_MO_DG_IR_and_opsets.html#intermediate_representation_used_in_openvino
[the guideline]:https://github.com/openvinotoolkit/openvino/wiki/BuildingForRaspbianStretchOS#cross-compilation-using-docker
[this guideline]:https://github.com/openvinotoolkit/open_model_zoo/blob/master/demos/README.md#build-the-demo-applications

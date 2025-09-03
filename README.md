# Repository for OpenVINO's extra modules

This repository is intended for the development of so-called "extra" modules, contributed functionality. New modules quite often do not have stable API, and they are not well-tested. Thus, they shouldn't be released as a part of official OpenVINO distribution, since the library maintains backward compatibility, and tries to provide decent performance and stability.

So, all the new modules should be developed separately, and published in the `openvino_contrib` repository at first. Later, when the module matures and gains popularity, it is moved to the central OpenVINO repository, and the development team provides production-quality support for this module.

## An overview of the openvino_contrib modules

This list gives an overview of all modules available inside the contrib repository.

* [**nvidia_plugin**](./modules/nvidia_plugin): NVIDIA GPU Plugin -- allows to perform deep neural networks inference on NVIDIA GPUs using CUDA, using OpenVINO API.
* [**java_api**](./modules/java_api): OpenVINO Java API -- provides Java wrappers for OpenVINO Runtime public API.
* [**rust_api**](https://github.com/intel/openvino-rs): Inference Engine Rust API -- explains how to use the OpenVINO APIs from Rust.
* [**custom_operations**](./modules/custom_operations/): Collection of Custom Operations -- implement Custom Operations with OpenVINO Extensibility Mechanism.
* [**Token Merging**](./modules/token_merging/): adaptation of [Token Merging method](https://arxiv.org/abs/2210.09461) for OpenVINO.
* [**OpenVINO Code**](./modules/openvino_code): VSCode extension for AI code completion with OpenVINO.
* [**Ollama-OpenVINO**](./modules/ollama_openvino): OpenVINO GenAI empowered Ollama which accelerate LLM on Intel platforms(including CPU, iGPU/dGPU, NPU).
* [**ov_training_kit**](./modules/ov_training_kit): Training Kit Python library -- provides scikit-learn, PyTorch and Tensorflow wrappers for training, optimization, and deployment with OpenVINO on AI PCs.

## How to build OpenVINO with extra modules
You can build OpenVINO, so it will include the modules from this repository. Contrib modules are under constant development and it is recommended to use them alongside the master branch or latest releases of OpenVINO.

Here is the CMake command for you:

```sh
$ cd <openvino_build_directory>
$ cmake -DOPENVINO_EXTRA_MODULES=<openvino_contrib>/modules <openvino_source_directory>
$ cmake --build . -j8
```

As the result, OpenVINO will be built in the <openvino_build_directory> with all modules from `openvino_contrib` repository. To disable specific modules, use CMake's `BUILD_<module_name>` boolean options. Like in this example:

```sh
$ cmake -DOPENVINO_EXTRA_MODULES=<openvino_contrib>/modules -DBUILD_java_api=OFF <openvino_source_directory>
```

Additional build instructions are available for the following modules:

* [**nvidia_plugin**](./modules/nvidia_plugin/README.md)
* [**custom_operations**](./modules/custom_operations/README.md)
* [**ollama_OpenVINO**](./modules/ollama_openvino)
* [**openvino-langchain**](./modules/openvino-langchain): LangChain.js integrations for OpenVINO™

## Update the repository documentation
In order to keep a clean overview containing all contributed modules, the following files need to be created/adapted:

* Update this file. Here, you add your module with a single line description.

* Add a README.md inside your own module folder. This README explains which functionality (separate functions) is available, explains in somewhat more detail what the module is expected to do. If any extra requirements are needed to build the module without problems, add them here also.

## How to Contribute

We welcome community contributions to the `openvino_contrib` repository. If you have an idea how to improve the modules, please share it with us.
All guidelines for contributing to the repository can be found [here](CONTRIBUTING.md).

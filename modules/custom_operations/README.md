# Collection of Custom Operations using OpenVINO Extensibility Mechanism

This module provides a guide and the implementation of a few custom operations in the Intel OpenVINO runtime using its [Extensibility Mechanism](https://docs.openvino.ai/latest/openvino_docs_Extensibility_UG_Intro.html).

There are some use cases when OpenVINO Custom Operations could be applicable:

* There is an ONNX model which contains an operation not supported by OpenVINO.
* You have a PyTorch model, which could be converted to ONNX, with an operation not supported by OpenVINO.
* You want to replace a subgraph for ONNX model with one custom operation which would be supported by OpenVINO.

More specifically, here we implement custom OpenVINO operations that add support for the following native PyTorch operation:

* [torch.fft](examples/fft)

Also, it contains the conversion extension `translate_sentencepiece_tokenizer` and the operation extension `SentencepieceTokenizer`
to add support for the tokenization part from TensorFlow [universal-sentence-encoder-multilingual](https://tfhub.dev/google/universal-sentence-encoder-multilingual/3) model.
The conversion extension changes the input format of the model. So the custom operation `SentencepieceTokenizer` expects 1D string tensor packed into the bitstream of the specific format.
For more information about the format, check the code for `SentencepieceTokenizer`.

And other custom operations introduced by third-party frameworks:

* [calculate_grid](examples/calculate_grid) and [sparse_conv](examples/sparse_conv) from [Open3D](https://github.com/isl-org/Open3D)
* [complex_mul](examples/complex_mul) from [DIRECT](https://github.com/NKI-AI/direct)

You can find more information about how to create and use OpenVINO Extensions to facilitate mapping of custom operations from framework model representation to OpenVINO representation [here](https://docs.openvino.ai/latest/openvino_docs_Extensibility_UG_Frontend_Extensions.html).


## Build custom OpenVINO operation extension library

The C++ code implementing the custom operation is in the `user_ie_extensions` directory. You'll have to build an "extension library" from this code so that it can be loaded at runtime. The steps below describe the build process:

1. Install [OpenVINO Runtime for C++](https://docs.openvino.ai/latest/openvino_docs_install_guides_install_dev_tools.html#for-c-developers).

2. Build the library:

```bash
cd openvino_contrib/modules/custom_operations
mkdir build && cd build
cmake ../ -DCMAKE_BUILD_TYPE=Release && cmake --build . --parallel 4
```

If you need to build only some operations specify them with the `-DCUSTOM_OPERATIONS` option:
```bash
cmake ../ -DCMAKE_BUILD_TYPE=Release -DCUSTOM_OPERATIONS="complex_mul;fft"
```

- Please note that [OpenCV](https://opencv.org/) installation is required to build an extension for the [fft](examples/fft) operation. Other extentions still can be built without OpenCV.

You also could build the extension library [while building OpenVINO](../../README.md).

## Load and use custom OpenVINO operation extension library

You can use the custom OpenVINO operations implementation by loading it into the OpenVINO `Core` object at runtime. Then, load the model from the ONNX file with the `read_model()` API. Here's how to do that in Python:

```python
from openvino.runtime import Core

# Create Core and register user extension
core = Core()
core.add_extension('/path/to/libuser_ov_extensions.so')

# Load model from .onnx file directly
model = core.read_model('model.onnx')
compiled_model = core.compile_model(model, 'CPU')
```

You also can get OpenVINO IR model with Model Optimizer, just use extra `--extension` flag to specify a path to custom extensions:

```bash
ovc model.onnx --extension /path/to/libuser_ov_extensions.so
```

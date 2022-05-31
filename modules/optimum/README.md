# OpenVINO&trade; Integration with Hugging Face Optimum

[![Test Optimum](https://github.com/openvinotoolkit/openvino_contrib/actions/workflows/test_optimum.yml/badge.svg?branch=master)](https://github.com/openvinotoolkit/openvino_contrib/actions/workflows/test_optimum.yml?query=branch%3Amaster)

This module is an extension for the [Hugging Face Optimum](https://github.com/huggingface/optimum) library and brings an [OpenVINO&trade;](https://github.com/openvinotoolkit/openvino) backend for [Hugging Face Transformers](https://github.com/huggingface/transformers) :hugs:.

This project provides APIs to enable the following tools for use with Hugging Face models:

* [OpenVINO Runtime](#openvino-runtime)
* [Neural Network Compression Framework (NNCF)](#nncf)

## Install

Supported Python versions: 3.7, 3.8, 3.9.

Install openvino-optimum runtime:
```bash
pip install openvino-optimum
```

or with all dependencies (`nncf` and `openvino-dev`):
```bash
pip install openvino-optimum[all]
```

Since this module is being actively developed, we recommend to install the latest version from Github, with:

```bash
pip install --upgrade "git+https://github.com/openvinotoolkit/openvino_contrib.git#egg=openvino-optimum&subdirectory=modules/optimum"
```

or with all dependencies:
```bash
pip install --upgrade "git+https://github.com/openvinotoolkit/openvino_contrib.git#egg=openvino-optimum[all]&subdirectory=modules/optimum"
```

To use PyTorch or TensorFlow models, these frameworks should be installed as well, for example with `pip install torch==1.9.*` or `pip install tensorflow==2.5.*`. To use TensorFlow models, openvino-dev is required. This can either be installed seperately with `pip install openvino-dev` or by choosing the _all dependencies_ option.

## OpenVINO Runtime

This module provides an inference API for Hugging Face models. It is possible to use PyTorch or TensorFlow models, or to use the native OpenVINO IR format (a pair of files `ov_model.xml` and `ov_model.bin`). When using PyTorch or TensorFlow models, openvino-optimum converts the model in the background, for use with OpenVINO Runtime.

To use the OpenVINO backend, import one of the `AutoModel` classes with an `OV` prefix. Specify a model name or local path in the `from_pretrained` method. When specifying a model name from [Hugging Face's Model Hub](https://huggingface.co/models), for example `bert-base-uncased`, the model will be downloaded and converted in the background. To load an OpenVINO IR file, `<name_or_path>` should be the directory that contains `ov_model.xml` and `ov_model.bin`. If this directory does not contain a configuration file, a `config` parameter should also be specified.

```python
from optimum.intel.openvino import OVAutoModel

# PyTorch trained model with OpenVINO backend
model = OVAutoModel.from_pretrained(<name_or_path>, from_pt=True)

# TensorFlow trained model with OpenVINO backend
model = OVAutoModel.from_pretrained(<name_or_path>, from_tf=True)

# Initialize a model from OpenVINO IR
from transformers import AutoConfig

config = AutoConfig.from_pretrained(model_name)
model = OVAutoModel.from_pretrained(<name_or_path>, config=config)

# Initialize with a model hosted in OpenVINO Model Server (requires providing config)
model = OVAutoModel.from_pretrained(<model_url>, inference_backend="ovms", config=<config>)
```

To save a model that was loaded from PyTorch or TensorFlow to OpenVINO's IR format, use the `save_pretrained()` method:

```python
model.save_pretrained(<model_directory>)
```

`ov_model.xml` and `ov_model.bin` will be saved in `model_directory`.

**Note:** `save_pretrained()` method will not work if model has been initialized with `inference_backend="ovms"`

For a complete example of how to do inference on a Hugging Face model with openvino-optimum, please check
out the [Fill-Mask demo](examples/masked_lm_demo.py)

## NNCF

To use the NNCF component, install openvino-optimum with the `[nncf]` or `[all]` extras:

```bash
pip install openvino-optimum[nncf]
```


[NNCF](https://github.com/openvinotoolkit/nncf) is used for model training with applying such features like quantization, pruning. To enable NNCF in your training pipeline do the following steps:

1. Import `NNCFAutoConfig`:

```python
from optimum.intel.nncf import NNCFAutoConfig
```

> **NOTE**: `NNCFAutoConfig` must be imported before `transformers` to make the magic work

2. Initialize an NNCF configuration object from a `.json` file:

```python
nncf_config = NNCFAutoConfig.from_json(training_args.nncf_config)
```

3. Pass the NNCF configuration to the `Trainer` object. For example:

```python
model = AutoModelForQuestionAnswering.from_pretrained(<name_or_path>)

...

trainer = QuestionAnsweringTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset if training_args.do_train else None,
    eval_dataset=eval_dataset if training_args.do_eval else None,
    eval_examples=eval_examples if training_args.do_eval else None,
    tokenizer=tokenizer,
    data_collator=data_collator,
    post_process_function=post_processing_function,
    compute_metrics=compute_metrics,
    nncf_config=nncf_config,
)
```

> **NOTE:** The NNCF module is independent from the Runtime module. The `model` class for NNCF should be a regular Transformers model, not an `OVAutoModel`.


Example config files can be found in the [nncf/configs directory](https://github.com/openvinotoolkit/openvino_contrib/tree/master/modules/optimum/optimum/intel/nncf/configs) in this repository.

Training [examples](https://github.com/huggingface/transformers/tree/master/examples/pytorch) can be found in the Transformers library. To use them with NNCF, modify the code to add `nncf_config` as outlined above, and add `--nncf_config` with the path to the NNCF config file when training your model. For example:

```python examples/pytorch/token-classification/run_ner.py --model_name_or_path bert-base-cased --dataset_name conll2003 --output_dir bert_base_cased_conll_int8 --do_train --do_eval --save_strategy epoch --evaluation_strategy epoch --nncf_config nncf_bert_config_conll.json```

More command line examples with Hugging Face demos can be found in the [NNCF repository](https://github.com/openvinotoolkit/nncf/tree/develop/third_party_integration/huggingface_transformers). Note that the installation steps and patching the repository are not necessary when using the NNCF integration in openvino-optimum.


See the [Changelog](https://github.com/openvinotoolkit/openvino_contrib/wiki/OpenVINO%E2%84%A2-Integration-with-Optimum*-Changelog) page for details about module development.

> *Other names and brands may be claimed as the property of others.

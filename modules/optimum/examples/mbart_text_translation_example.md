# MBart Text Translation Example

In this example we will translate a sentance from one language to another using MBart model.
There are a few variants of working with the models possible and here we will:

1. [Work with model loaded from Hugging Face](#work-with-model-loaded-from-hugging-face)
2. [Work with OpenVINO model hosted in OpenVINO Model Server](#work-with-openvino-model-hosted-in-openvino-model-server)

... but first install OpenVINO Optimum python package as described in the section below.
## Install OpenVINO Optimum

Please clone this repository and install OpenVINO Optimum python package and it's dependencies:

```bash
git clone https://github.com/mzegla/openvino_contrib.git
cd openvino_contrib/modules/optimum/
git checkout optimum-adapters
pip install --upgrade pip
pip install .
pip install torch==1.9.1
pip install sentencepiece
```

## Work with model loaded from Hugging Face

```python
from optimum.intel.openvino import OVMBartForConditionalGeneration
from transformers import MBart50TokenizerFast

model = OVMBartForConditionalGeneration.from_pretrained("dkurt/mbart-large-50-many-to-many-mmt-int8")
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

# Input is a sentence in Polish and we will translate it to English
input_sentence = "Witaj świecie!"
tokenizer.src_lang = "pl_PL"
encoded_input_sentence = tokenizer(input_sentence, return_tensors="pt")
generated_tokens = model.generate(**encoded_input_sentence, forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"])
decoded_output = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
# Print English translation of the Polish input
print(decoded_output)
```

## Work with OpenVINO model hosted in OpenVINO Model Server

In this variant the inference is being delegated to the OpenVINO Model Server.
Running MBart with OVMS requires more manual steps, configuration and some additional operations. 
That's why we will use a separate method that will prepare OpenVINO Model Server image with MBart already included and configured to run with Optimum.
To do that first install `docker` package in your Python environment (note that you also must have Docker engine installed and running on the host).

 - Install [Docker engine](https://docs.docker.com/engine/install/) on the machine you'll run OpenVINO Model Sever on.
 - Run `pip install docker` to install Docker SDK on your Python environment

Once you have it, run the following Python code:

```python
from optimum.intel.openvino import OVMBartForConditionalGeneration
model = OVMBartForConditionalGeneration.from_pretrained("dkurt/mbart-large-50-many-to-many-mmt-int8")
model.create_ovms_image("ovms_mbart:latest")
```

Then use the newly created image to start a container. Now to make it work you just need to specify the gRPC port that OVMS will listen on and forward it to the container.

```bash
docker run --rm -d -p 9999:9999 ovms_mbart:latest --port 9999
```

Now that OpenVINO Model Server is running, let's run below sample to translate the sentence with MBart:


```python
from optimum.intel.openvino import OVMBartForConditionalGeneration
from transformers import MBart50TokenizerFast, AutoConfig

config = AutoConfig.from_pretrained("dkurt/mbart-large-50-many-to-many-mmt-int8")
model = OVMBartForConditionalGeneration.from_pretrained(["localhost:9999/models/encoder","localhost:9999/models/model", "localhost:9999/models/model_past"], inference_backend="ovms", config=config)

tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

# Input is a sentence in Polish and we will translate it to English
input_sentence = "Witaj świecie!"
tokenizer.src_lang = "pl_PL"
encoded_input_sentence = tokenizer(input_sentence, return_tensors="pt")
generated_tokens = model.generate(**encoded_input_sentence, forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"])
decoded_output = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
# Print English translation of the Polish input
print(decoded_output)
```

## Summary

If you look into the code, all the variants are almost identical. The main difference is the way `OVMBartForConditionalGeneration` object gets created. Depending on how you want to load and use the model you can choose to download original model from Hugging Face and run the optimization implicitly, load local OpenVINO model, or use the model hosted remotely in OpenVINO Model Server. Most of the code remains the same no matter which option you've chosen.
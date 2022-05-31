# BERT Question Answering Example

In this example we will run question answering on BERT model.
There are a few variants of working with the models possible and here we will:

1. [Work with model loaded from Hugging Face](#work-with-model-loaded-from-hugging-face)
2. [Work with OpenVINO model saved on local filesystem](#work-with-openvino-model-saved-on-local-filesystem)
3. [Work with OpenVINO model hosted in OpenVINO Model Server](#work-with-openvino-model-hosted-in-openvino-model-server)

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
```

## Work with model loaded from Hugging Face

```python
import numpy as np
from transformers import AutoTokenizer, AutoConfig
from optimum.intel.openvino import OVAutoModelForQuestionAnswering

warmup_question = "What is pi?"
context = "The number π (/paɪ/; spelled out as 'pi') is a mathematical constant that is the ratio of a circle's circumference to its diameter"

model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"

# Load model from Hugging Face (and run model optimization on original PyTorch model)
model = OVAutoModelForQuestionAnswering.from_pretrained(
    model_name, from_pt=True
) 

tokenizer = AutoTokenizer.from_pretrained(model_name)
warmup_input = tokenizer.encode_plus(
    warmup_question, context, return_tensors="np", add_special_tokens=True
)

result = model(**warmup_input, return_dict=True)
answer_start_scores = result["start_logits"]
answer_end_scores = result["end_logits"]

input_ids = warmup_input["input_ids"].tolist()[0]
answer_start = np.argmax(answer_start_scores)
answer_end = np.argmax(answer_end_scores) + 1
answer = tokenizer.convert_tokens_to_string(
    tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end])
)
print(f"Question: {warmup_question}")
print(f"Answer: {answer}")

# save model locally (for the next variant)
model_dir = f"{model_name}_fp32"
model.save_pretrained(model_dir)
```

## Work with OpenVINO model saved on local filesystem

The last lines of previous sample saved OpenVINO model in IR format on the local filesystem. For faster execution we will use locally saved model instead of pulling from Hugging Face:

```python
import numpy as np
from transformers import AutoTokenizer, AutoConfig
from optimum.intel.openvino import OVAutoModelForQuestionAnswering

warmup_question = "What is pi?"
context = "The number π (/paɪ/; spelled out as 'pi') is a mathematical constant that is the ratio of a circle's circumference to its diameter"

model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"

config = AutoConfig.from_pretrained(model_name)

# Load model from a directory with XML and BIN files of OpenVINO optimized model
model_dir = f"{model_name}_fp32"
model = OVAutoModelForQuestionAnswering.from_pretrained(
     model_dir, config=config
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
warmup_input = tokenizer.encode_plus(
    warmup_question, context, return_tensors="np", add_special_tokens=True
)

result = model(**warmup_input, return_dict=True)
answer_start_scores = result["start_logits"]
answer_end_scores = result["end_logits"]

input_ids = warmup_input["input_ids"].tolist()[0]
answer_start = np.argmax(answer_start_scores)
answer_end = np.argmax(answer_end_scores) + 1
answer = tokenizer.convert_tokens_to_string(
    tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end])
)
print(f"Question: {warmup_question}")
print(f"Answer: {answer}")
```

## Work with OpenVINO model hosted in OpenVINO Model Server

In this variant the inference is being delegated to the OpenVINO Model Server.
Before running the sample we need to put OpenVINO model used in previous step in a directory structure accepted by the model server:

```bash
mkdir -p bert/1
cp -R bert-large-uncased-whole-word-masking-finetuned-squad_fp32/ov_model.* bert/1
```

Then launch OpenVINO Model Server with BERT model and configure it to accept dynamic input shapes:

```bash
docker run --rm -d -p 9000:9000 -v $PWD/bert:/bert openvino/model_server:latest --model_name bert --model_path /bert --port 9000 --shape "{\"input_ids\": \"(-1,-1)\", \"attention_mask\": \"(-1,-1)\"}"
```

Now that OpenVINO Model Server is running, let's run below sample to perform question answering just like in the variants presented earlier:


```python
import numpy as np
from transformers import AutoTokenizer, AutoConfig
from optimum.intel.openvino import OVAutoModelForQuestionAnswering

warmup_question = "What is pi?"
context = "The number π (/paɪ/; spelled out as 'pi') is a mathematical constant that is the ratio of a circle's circumference to its diameter"

model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"

config = AutoConfig.from_pretrained(model_name)

# Use model called "bert" hosted on port 9000 with OpenVINO Model Server (OVMS)
model = OVAutoModelForQuestionAnswering.from_pretrained(
     "localhost:9000/models/bert", inference_backend="ovms", config=config
)


tokenizer = AutoTokenizer.from_pretrained(model_name)
warmup_input = tokenizer.encode_plus(
    warmup_question, context, return_tensors="np", add_special_tokens=True
)

result = model(**warmup_input, return_dict=True)
answer_start_scores = result["start_logits"]
answer_end_scores = result["end_logits"]

input_ids = warmup_input["input_ids"].tolist()[0]
answer_start = np.argmax(answer_start_scores)
answer_end = np.argmax(answer_end_scores) + 1
answer = tokenizer.convert_tokens_to_string(
    tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end])
)
print(f"Question: {warmup_question}")
print(f"Answer: {answer}")
```

## Summary

If you look into the code, all the variants are almost identical. The main difference is the way `OVAutoModelForQuestionAnswering` object gets created. Depending on how you want to load and use the model you can choose to download original model from Hugging Face and run the optimization implicitly, load local OpenVINO model, or use the model hosted remotely in OpenVINO Model Server. Most of the code remains the same no matter which option you've chosen.
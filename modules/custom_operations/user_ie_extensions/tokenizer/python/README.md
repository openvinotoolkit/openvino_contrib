# OpenVINO Tokenizers

## Features

- Convert a HuggingFace tokenizer into OpenVINO model tokenizer and detokenizer:
  - Fast tokenizers based on Wordpiece and BPE models
  - Slow tokenizers based on SentencePiece model file
- Combine OpenVINO models into a single model

## Installation

1. Build the extension with the `-DCUSTOM_OPERATIONS="tokenizer"` flag: [instruction](../../../README.md#build-custom-openvino-operation-extension-library)
2. (Recommended) Create and activate virtual env:
```bash
python3 -m venv venv
source venv/bin/activate
```
3. Go to `modules/custom_operations/user_ie_extensions/tokenizer/python` and run:
```bash
# to use converted tokenizers or models combined with tokenizers
pip install .
# to convert tokenizers from transformers library
pip install .[transformers] 
# for development and testing the library
pip isntall -e .[all]
```

## Usage

Set `OV_TOKENIZER_PREBUILD_EXTENSION_PATH` environment variable to `libuser_ov_extensions.so` file path
or use `init_extension` function.

### Convert HuggingFace tokenizer

```python
from transformers import AutoTokenizer
from openvino import compile_model
from ov_tokenizer import init_extension, convert_tokenizer, pack_strings


init_extension("path/to/libuser_ov_extensions.so")

hf_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
ov_tokenizer = convert_tokenizer(hf_tokenizer)

compiled_tokenzier = compile_model(ov_tokenizer)
text_input = "Test string"

hf_output = hf_tokenizer([text_input], return_tensors="np")
ov_output = compiled_tokenzier(pack_strings([text_input]))

for output_name in hf_output:
    print(f"OpenVINO {output_name} = {ov_output[output_name]}")
    print(f"HuggingFace {output_name} = {hf_output[output_name]}")
# OpenVINO input_ids = [[ 101 3231 5164  102]]
# HuggingFace input_ids = [[ 101 3231 5164  102]]
# OpenVINO token_type_ids = [[0 0 0 0]]
# HuggingFace token_type_ids = [[0 0 0 0]]
# OpenVINO attention_mask = [[1 1 1 1]]
# HuggingFace attention_mask = [[1 1 1 1]]
```

### Connect Tokenizer to a Model


```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from openvino import compile_model, convert_model
from ov_tokenizer import init_extension, convert_tokenizer, pack_strings, connect_models


init_extension("path/to/libuser_ov_extensions.so")

checkpoint = "mrm8488/bert-tiny-finetuned-sms-spam-detection"
hf_tokenizer = AutoTokenizer.from_pretrained(checkpoint)
hf_model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

text_input = ["Free money!!!"]
hf_input = hf_tokenizer(text_input, return_tensors="pt")
hf_output = hf_model(**hf_input)

ov_tokenizer = convert_tokenizer(hf_tokenizer)
ov_model = convert_model(hf_model, example_input=hf_input.data)
combined_model = connect_models(ov_tokenizer, ov_model)
compiled_combined_model = compile_model(combined_model)

openvino_output = compiled_combined_model(pack_strings(text_input))

print(f"OpenVINO logits: {openvino_output['logits']}")
# OpenVINO logits: [[ 1.2007061 -1.4698029]]
print(f"HuggingFace logits {hf_output.logits}")
# HuggingFace logits tensor([[ 1.2007, -1.4698]], grad_fn=<AddmmBackward0>)
```

### Convert SentencePiece Model Tokenzier

```python
from transformers import AutoTokenizer
from openvino import compile_model
from ov_tokenizer import init_extension, convert_sentencepiece_model_tokenizer, pack_strings, unpack_strings


init_extension("path/to/libuser_ov_extensions.so")

checkpoint = "codellama/CodeLlama-7b-hf"
hf_tokenizer = AutoTokenizer.from_pretrained(checkpoint)

text_input = ["def fibonnaci(n):"]
hf_input = hf_tokenizer(text_input, return_tensors="np")

ov_tokenizer, ov_detokenizer = convert_sentencepiece_model_tokenizer(hf_tokenizer, with_decoder=True)
compiled_tokenizer = compile_model(ov_tokenizer)
compiled_detokenizer = compile_model(ov_detokenizer)
ov_input = compiled_tokenizer(pack_strings(text_input))

for model_input_name in hf_input:
    print(f"OpenVINO {model_input_name} = {ov_input[model_input_name]}")
    print(f"HuggingFace {model_input_name} = {hf_input[model_input_name]}")
# OpenVINO input_ids = [[    1   822 18755 11586   455 29898 29876  1125]]
# HuggingFace input_ids = [[    1   822 18755 11586   455 29898 29876  1125]]
# OpenVINO attention_mask = [[1 1 1 1 1 1 1 1]]
# HuggingFace attention_mask = [[1 1 1 1 1 1 1 1]]
    
ov_output = unpack_strings(compiled_detokenizer(hf_input.input_ids)["string_output"])
hf_output = hf_tokenizer.batch_decode(hf_input.input_ids, skip_special_tokens=True)
print(f"OpenVINO output string: `{ov_output}`")
# OpenVINO output string: ['def fibonnaci(n):']
print(f"HuggingFace output string: `{hf_output}`")
# HuggingFace output string: ['def fibonnaci(n):']
```

To connect a detokenizer to a `logits` model output, set `greedy_decoder=True` when using the `convert_tokenizer` or `convert_sentencepiece_model_tokenizer` function, enabling a greedy decoding pipeline before detoknizer. This allows the detokenizer to be connected to the `logits` model output.

### Use Extension With Converted (De)Tokenizer or Model combined with (De)Tokenizer

To work with converted tokenizer you need `pack_strings`/`unpack_strings` functions. 

```python
import numpy as np
from openvino import Core
from ov_tokenizer import unpack_strings


core = Core()
core.add_extension("path/to/libuser_ov_extensions.so")
# detokenizer from codellama sentencepiece model
compiled_detokenizer = core.compile_model("detokenizer.xml")

token_ids = np.random.randint(100, 1000, size=(3, 5))
openvino_output = compiled_detokenizer(token_ids)

print(unpack_strings(openvino_output["string_output"]))
# ['sc�ouition�', 'intvenord hasient', 'g shouldwer M more']
```

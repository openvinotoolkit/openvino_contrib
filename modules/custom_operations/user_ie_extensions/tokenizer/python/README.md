# OpenVINO Tokenizers

## Features

- Convert a HuggingFace tokenizer into OpenVINO model tokenizer and detokenizer:
  - Fast tokenizers based on Wordpiece and BPE models
  - Slow tokenizers based on SentencePiece model file
- Combine OpenVINO models into a single model
- Add greedy decoding pipeline to text generation model

## Installation

1. Install [OpenVINO Runtime for C++](https://docs.openvino.ai/latest/openvino_docs_install_guides_install_dev_tools.html#for-c-developers).
2. (Recommended) Create and activate virtual env:
```bash
python3 -m venv venv
source venv/bin/activate
```
3. Go to `modules/custom_operations` and run:
```bash
# to use converted tokenizers or models combined with tokenizers
pip install .
# to convert tokenizers from transformers library
pip install .[transformers]
# for development and testing the library
pip install -e .[all]
```

### Convert HuggingFace tokenizer

```python
from transformers import AutoTokenizer
from openvino import compile_model
from openvino_tokenizers import convert_tokenizer
import numpy as np  # TODO: Remove after OV PythonAPI will support list arguments

hf_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
ov_tokenizer = convert_tokenizer(hf_tokenizer)

compiled_tokenzier = compile_model(ov_tokenizer)
text_input = "Test string"

hf_output = hf_tokenizer([text_input], return_tensors="np")
ov_output = compiled_tokenzier(np.array([text_input]))  # TODO: Remove np.array after OV PythonAPI will support list arguments

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
from openvino_tokenizers import convert_tokenizer, connect_models
import numpy as np  # TODO: Remove after OV PythonAPI will support list arguments

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

openvino_output = compiled_combined_model(np.array(text_input))  # TODO: Remove np.array after OV PythonAPI will support list arguments

print(f"OpenVINO logits: {openvino_output['logits']}")
# OpenVINO logits: [[ 1.2007061 -1.4698029]]
print(f"HuggingFace logits {hf_output.logits}")
# HuggingFace logits tensor([[ 1.2007, -1.4698]], grad_fn=<AddmmBackward0>)
```

### Use Extension With Converted (De)Tokenizer or Model With (De)Tokenizer

To work with converted tokenizer and detokenizer, numpy string tensors are used.

```python
import numpy as np
from openvino import Core

core = Core()

# detokenizer from codellama sentencepiece model
compiled_detokenizer = core.compile_model("detokenizer.xml")

token_ids = np.random.randint(100, 1000, size=(3, 5))
openvino_output = compiled_detokenizer(token_ids)

print(openvino_output["string_output"])
# ['sc�ouition�', 'intvenord hasient', 'g shouldwer M more']
```

### Text generation pipeline

```python
import numpy as np
from openvino import compile_model, convert_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from openvino_tokenizers import add_greedy_decoding, convert_tokenizer

# Use different repo for the tokenizer because the original repo doesn't have .model file
# Sentencepiece(Unigram) tokenizer supported only with .model file
tokenizer_checkpoint = "microsoft/Llama2-7b-WhoIsHarryPotter"
model_checkpoint = "nickypro/tinyllama-15M"
hf_tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)
hf_model = AutoModelForCausalLM.from_pretrained(model_checkpoint, use_cache=False)

# convert hf tokenizer
text_input = ["Quick brown fox was"]
ov_tokenizer, ov_detokenizer = convert_tokenizer(hf_tokenizer, with_detokenizer=True)
compiled_tokenizer = compile_model(ov_tokenizer)

# transform input text into tokens
ov_input = compiled_tokenizer(np.array(text_input))  # TODO: Remove np.array after OV PythonAPI will support list arguments
hf_input = hf_tokenizer(text_input, return_tensors="pt")

# convert Pytorch model to OpenVINO IR and add greedy decoding pipeline to it
ov_model = convert_model(hf_model, example_input=hf_input.data)
ov_model_with_greedy_decoding = add_greedy_decoding(ov_model)
compiled_model = compile_model(ov_model_with_greedy_decoding)

# generate new tokens
new_tokens_size = 10
prompt_size = ov_input["input_ids"].shape[-1]
input_dict = {
  output.any_name: np.hstack([tensor, np.zeros(shape=(1, new_tokens_size), dtype=np.int_)])
  for output, tensor in ov_input.items()
}
for idx in range(prompt_size, prompt_size + new_tokens_size):
  output = compiled_model(input_dict)["token_ids"]
  input_dict["input_ids"][:, idx] = output[:, idx - 1]
  input_dict["attention_mask"][:, idx] = 1
ov_token_ids = input_dict["input_ids"]

hf_token_ids = hf_model.generate(
  **hf_input,
  min_new_tokens=new_tokens_size,
  max_new_tokens=new_tokens_size,
  temperature=0,  # greedy decoding
)

# decode model output
compiled_detokenizer = compile_model(ov_detokenizer)
ov_output = compiled_detokenizer(ov_token_ids)["string_output"]
hf_output = hf_tokenizer.batch_decode(hf_token_ids, skip_special_tokens=True)
print(f"OpenVINO output string: `{ov_output}`")
# OpenVINO output string: `['Quick brown fox was walking through the forest. He was looking for something']`
print(f"HuggingFace output string: `{hf_output}`")
# HuggingFace output string: `['Quick brown fox was walking through the forest. He was looking for something']`
```

## Test Coverage

This report is autogenerated and includes tokenizers and detokenizers tests. To update it run pytest with `--update_readme` flag.

### Coverage by Tokenizer Type

<table>
  <thead>
    <tr>
      <th >Tokenizer Type</th>
      <th >Pass Rate, %</th>
      <th >Number of Tests</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td >BPE</td>
      <td >91.34</td>
      <td >1190</td>
    </tr>
    <tr>
      <td >SentencePiece</td>
      <td >61.25</td>
      <td >1120</td>
    </tr>
    <tr>
      <td >Tiktoken</td>
      <td >96.43</td>
      <td >140</td>
    </tr>
    <tr>
      <td >WordPiece</td>
      <td >86.79</td>
      <td >507</td>
    </tr>
  </tbody>
</table>

### Coverage by Model Type

<table>
  <thead>
    <tr>
      <th >Tokenizer Type</th>
      <th >Model</th>
      <th >Pass Rate, %</th>
      <th >Number of Tests</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td >BPE</td>
      <td >EleutherAI/gpt-j-6b</td>
      <td >95.71</td>
      <td >70</td>
    </tr>
    <tr>
      <td >BPE</td>
      <td >EleutherAI/gpt-neo-125m</td>
      <td >95.71</td>
      <td >70</td>
    </tr>
    <tr>
      <td >BPE</td>
      <td >EleutherAI/gpt-neox-20b</td>
      <td >94.29</td>
      <td >70</td>
    </tr>
    <tr>
      <td >BPE</td>
      <td >EleutherAI/pythia-12b-deduped</td>
      <td >94.29</td>
      <td >70</td>
    </tr>
    <tr>
      <td >BPE</td>
      <td >KoboldAI/fairseq-dense-13B</td>
      <td >97.14</td>
      <td >70</td>
    </tr>
    <tr>
      <td >BPE</td>
      <td >Salesforce/codegen-16B-multi</td>
      <td >92.86</td>
      <td >70</td>
    </tr>
    <tr>
      <td >BPE</td>
      <td >ai-forever/rugpt3large_based_on_gpt2</td>
      <td >94.29</td>
      <td >70</td>
    </tr>
    <tr>
      <td >BPE</td>
      <td >bigscience/bloom</td>
      <td >98.57</td>
      <td >70</td>
    </tr>
    <tr>
      <td >BPE</td>
      <td >facebook/bart-large-mnli</td>
      <td >92.86</td>
      <td >70</td>
    </tr>
    <tr>
      <td >BPE</td>
      <td >facebook/galactica-120b</td>
      <td >95.71</td>
      <td >70</td>
    </tr>
    <tr>
      <td >BPE</td>
      <td >facebook/opt-66b</td>
      <td >97.14</td>
      <td >70</td>
    </tr>
    <tr>
      <td >BPE</td>
      <td >gpt2</td>
      <td >92.86</td>
      <td >70</td>
    </tr>
    <tr>
      <td >BPE</td>
      <td >laion/CLIP-ViT-bigG-14-laion2B-39B-b160k</td>
      <td >47.14</td>
      <td >70</td>
    </tr>
    <tr>
      <td >BPE</td>
      <td >microsoft/deberta-base</td>
      <td >90.00</td>
      <td >70</td>
    </tr>
    <tr>
      <td >BPE</td>
      <td >roberta-base</td>
      <td >90.00</td>
      <td >70</td>
    </tr>
    <tr>
      <td >BPE</td>
      <td >sentence-transformers/all-roberta-large-v1</td>
      <td >88.57</td>
      <td >70</td>
    </tr>
    <tr>
      <td >BPE</td>
      <td >stabilityai/stablecode-completion-alpha-3b-4k</td>
      <td >95.71</td>
      <td >70</td>
    </tr>
    <tr>
      <td >SentencePiece</td>
      <td >NousResearch/Llama-2-13b-hf</td>
      <td >100.00</td>
      <td >70</td>
    </tr>
    <tr>
      <td >SentencePiece</td>
      <td >NousResearch/Llama-2-13b-hf_slow</td>
      <td >100.00</td>
      <td >70</td>
    </tr>
    <tr>
      <td >SentencePiece</td>
      <td >THUDM/chatglm2-6b</td>
      <td >50.00</td>
      <td >70</td>
    </tr>
    <tr>
      <td >SentencePiece</td>
      <td >THUDM/chatglm2-6b_slow</td>
      <td >50.00</td>
      <td >70</td>
    </tr>
    <tr>
      <td >SentencePiece</td>
      <td >THUDM/chatglm3-6b</td>
      <td >100.00</td>
      <td >70</td>
    </tr>
    <tr>
      <td >SentencePiece</td>
      <td >THUDM/chatglm3-6b_slow</td>
      <td >100.00</td>
      <td >70</td>
    </tr>
    <tr>
      <td >SentencePiece</td>
      <td >camembert-base</td>
      <td >25.71</td>
      <td >70</td>
    </tr>
    <tr>
      <td >SentencePiece</td>
      <td >camembert-base_slow</td>
      <td >25.71</td>
      <td >70</td>
    </tr>
    <tr>
      <td >SentencePiece</td>
      <td >codellama/CodeLlama-7b-hf</td>
      <td >100.00</td>
      <td >70</td>
    </tr>
    <tr>
      <td >SentencePiece</td>
      <td >codellama/CodeLlama-7b-hf_slow</td>
      <td >100.00</td>
      <td >70</td>
    </tr>
    <tr>
      <td >SentencePiece</td>
      <td >microsoft/deberta-v3-base</td>
      <td >85.71</td>
      <td >70</td>
    </tr>
    <tr>
      <td >SentencePiece</td>
      <td >microsoft/deberta-v3-base_slow</td>
      <td >97.14</td>
      <td >70</td>
    </tr>
    <tr>
      <td >SentencePiece</td>
      <td >xlm-roberta-base</td>
      <td >0.00</td>
      <td >70</td>
    </tr>
    <tr>
      <td >SentencePiece</td>
      <td >xlm-roberta-base_slow</td>
      <td >0.00</td>
      <td >70</td>
    </tr>
    <tr>
      <td >SentencePiece</td>
      <td >xlnet-base-cased</td>
      <td >22.86</td>
      <td >70</td>
    </tr>
    <tr>
      <td >SentencePiece</td>
      <td >xlnet-base-cased_slow</td>
      <td >22.86</td>
      <td >70</td>
    </tr>
    <tr>
      <td >Tiktoken</td>
      <td >Qwen/Qwen-14B-Chat</td>
      <td >97.14</td>
      <td >70</td>
    </tr>
    <tr>
      <td >Tiktoken</td>
      <td >Salesforce/xgen-7b-8k-base</td>
      <td >95.71</td>
      <td >70</td>
    </tr>
    <tr>
      <td >WordPiece</td>
      <td >ProsusAI/finbert</td>
      <td >84.62</td>
      <td >39</td>
    </tr>
    <tr>
      <td >WordPiece</td>
      <td >bert-base-multilingual-cased</td>
      <td >84.62</td>
      <td >39</td>
    </tr>
    <tr>
      <td >WordPiece</td>
      <td >bert-large-cased</td>
      <td >84.62</td>
      <td >39</td>
    </tr>
    <tr>
      <td >WordPiece</td>
      <td >cointegrated/rubert-tiny2</td>
      <td >84.62</td>
      <td >39</td>
    </tr>
    <tr>
      <td >WordPiece</td>
      <td >distilbert-base-uncased-finetuned-sst-2-english</td>
      <td >84.62</td>
      <td >39</td>
    </tr>
    <tr>
      <td >WordPiece</td>
      <td >google/electra-base-discriminator</td>
      <td >84.62</td>
      <td >39</td>
    </tr>
    <tr>
      <td >WordPiece</td>
      <td >google/mobilebert-uncased</td>
      <td >100.00</td>
      <td >39</td>
    </tr>
    <tr>
      <td >WordPiece</td>
      <td >jhgan/ko-sbert-sts</td>
      <td >79.49</td>
      <td >39</td>
    </tr>
    <tr>
      <td >WordPiece</td>
      <td >prajjwal1/bert-mini</td>
      <td >100.00</td>
      <td >39</td>
    </tr>
    <tr>
      <td >WordPiece</td>
      <td >rajiv003/ernie-finetuned-qqp</td>
      <td >100.00</td>
      <td >39</td>
    </tr>
    <tr>
      <td >WordPiece</td>
      <td >rasa/LaBSE</td>
      <td >76.92</td>
      <td >39</td>
    </tr>
    <tr>
      <td >WordPiece</td>
      <td >sentence-transformers/all-MiniLM-L6-v2</td>
      <td >79.49</td>
      <td >39</td>
    </tr>
    <tr>
      <td >WordPiece</td>
      <td >squeezebert/squeezebert-uncased</td>
      <td >84.62</td>
      <td >39</td>
    </tr>
  </tbody>
</table>

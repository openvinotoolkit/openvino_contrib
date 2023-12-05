# OpenVINO Tokenizers

## Features

- Convert a HuggingFace tokenizer into OpenVINO model tokenizer and detokenizer:
  - Fast tokenizers based on Wordpiece and BPE models
  - Slow tokenizers based on SentencePiece model file
- Combine OpenVINO models into a single model
- Add greedy decoding pipeline to text generation model 

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

### Use Extension With Converted (De)Tokenizer or Model With (De)Tokenizer

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

### Text generation pipeline

```python
import numpy as np
from openvino import compile_model, convert_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from ov_tokenizer import (
    add_greedy_decoding,
    convert_tokenizer,
    init_extension,
    pack_strings,
    unpack_strings,
)


init_extension("path/to/libuser_ov_extensions.so")

# Use different repo for the tokenizer because the original repo doesn't have .model file
# Sentencepiece(Unigram) tokenizer supported only with .model file
tokenizer_checkpoint = "microsoft/Llama2-7b-WhoIsHarryPotter"
model_checkpoint = "nickypro/tinyllama-15M"
hf_tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)
hf_model = AutoModelForCausalLM.from_pretrained(model_checkpoint, use_cache=False)

# convert hf tokenizer
text_input = ["Quick brown fox was"]
ov_tokenizer, ov_detokenizer = convert_tokenizer(hf_tokenizer, with_decoder=True)
compiled_tokenizer = compile_model(ov_tokenizer)

# transform input text into tokens
ov_input = compiled_tokenizer(pack_strings(text_input))
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
ov_output = unpack_strings(compiled_detokenizer(ov_token_ids)["string_output"])
hf_output = hf_tokenizer.batch_decode(hf_token_ids, skip_special_tokens=True)
print(f"OpenVINO output string: `{ov_output}`")
# OpenVINO output string: `['Quick brown fox was walking through the forest. He was looking for something']`
print(f"HuggingFace output string: `{hf_output}`")
# HuggingFace output string: `['Quick brown fox was walking through the forest. He was looking for something']`
```
## Test Coverage

### Covarage by Tokenizer Type
<style type="text/css">
#T_08334_row0_col0 {
  background-color: #3885bc;
  color: #f1f1f1;
}
#T_08334_row1_col0 {
  background-color: #67001f;
  color: #f1f1f1;
}
#T_08334_row2_col0 {
  background-color: #246aae;
  color: #f1f1f1;
}
#T_08334_row3_col0 {
  background-color: #053061;
  color: #f1f1f1;
}
</style>
<table id="T_08334_">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th class="col_heading level0 col0" >status</th>
    </tr>
    <tr>
      <th class="index_name level0" >tokenizer_type</th>
      <th class="blank col0" >&nbsp;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_08334_level0_row0" class="row_heading level0 row0" >bpe</th>
      <td id="T_08334_row0_col0" class="data row0 col0" >0.929412</td>
    </tr>
    <tr>
      <th id="T_08334_level0_row1" class="row_heading level0 row1" >sentencepiece</th>
      <td id="T_08334_row1_col0" class="data row1 col0" >0.620000</td>
    </tr>
    <tr>
      <th id="T_08334_level0_row2" class="row_heading level0 row2" >tiktoken</th>
      <td id="T_08334_row2_col0" class="data row2 col0" >0.950000</td>
    </tr>
    <tr>
      <th id="T_08334_level0_row3" class="row_heading level0 row3" >wordpiece</th>
      <td id="T_08334_row3_col0" class="data row3 col0" >0.992042</td>
    </tr>
  </tbody>
</table>

 ### Covarage by Model Type
<style type="text/css">
#T_fd076_row0_col0, #T_fd076_row1_col0, #T_fd076_row4_col0, #T_fd076_row8_col0, #T_fd076_row10_col0, #T_fd076_row11_col0, #T_fd076_row13_col0, #T_fd076_row14_col0, #T_fd076_row15_col0, #T_fd076_row33_col0 {
  background-color: #10457e;
  color: #f1f1f1;
}
#T_fd076_row2_col0, #T_fd076_row3_col0, #T_fd076_row5_col0 {
  background-color: #1b5a9c;
  color: #f1f1f1;
}
#T_fd076_row6_col0, #T_fd076_row9_col0, #T_fd076_row16_col0, #T_fd076_row27_col0, #T_fd076_row34_col0 {
  background-color: #15508d;
  color: #f1f1f1;
}
#T_fd076_row7_col0, #T_fd076_row28_col0 {
  background-color: #0a3b70;
  color: #f1f1f1;
}
#T_fd076_row12_col0 {
  background-color: #d1e5f0;
  color: #000000;
}
#T_fd076_row17_col0, #T_fd076_row18_col0, #T_fd076_row21_col0, #T_fd076_row22_col0, #T_fd076_row25_col0, #T_fd076_row26_col0, #T_fd076_row35_col0, #T_fd076_row36_col0, #T_fd076_row37_col0, #T_fd076_row38_col0, #T_fd076_row39_col0, #T_fd076_row40_col0, #T_fd076_row41_col0, #T_fd076_row42_col0, #T_fd076_row43_col0, #T_fd076_row44_col0, #T_fd076_row46_col0, #T_fd076_row47_col0 {
  background-color: #053061;
  color: #f1f1f1;
}
#T_fd076_row19_col0, #T_fd076_row20_col0 {
  background-color: #f6f7f7;
  color: #000000;
}
#T_fd076_row23_col0, #T_fd076_row24_col0 {
  background-color: #e8896c;
  color: #f1f1f1;
}
#T_fd076_row29_col0, #T_fd076_row30_col0 {
  background-color: #67001f;
  color: #f1f1f1;
}
#T_fd076_row31_col0, #T_fd076_row32_col0 {
  background-color: #e27b62;
  color: #f1f1f1;
}
#T_fd076_row45_col0 {
  background-color: #2267ac;
  color: #f1f1f1;
}
</style>
<table id="T_fd076_">
  <thead>
    <tr>
      <th class="blank" >&nbsp;</th>
      <th class="blank level0" >&nbsp;</th>
      <th class="col_heading level0 col0" >status</th>
    </tr>
    <tr>
      <th class="index_name level0" >tokenizer_type</th>
      <th class="index_name level1" >models</th>
      <th class="blank col0" >&nbsp;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_fd076_level0_row0" class="row_heading level0 row0" rowspan="17">bpe</th>
      <th id="T_fd076_level1_row0" class="row_heading level1 row0" >EleutherAI/gpt-j-6b</th>
      <td id="T_fd076_row0_col0" class="data row0 col0" >0.960000</td>
    </tr>
    <tr>
      <th id="T_fd076_level1_row1" class="row_heading level1 row1" >EleutherAI/gpt-neo-125m</th>
      <td id="T_fd076_row1_col0" class="data row1 col0" >0.960000</td>
    </tr>
    <tr>
      <th id="T_fd076_level1_row2" class="row_heading level1 row2" >EleutherAI/gpt-neox-20b</th>
      <td id="T_fd076_row2_col0" class="data row2 col0" >0.920000</td>
    </tr>
    <tr>
      <th id="T_fd076_level1_row3" class="row_heading level1 row3" >EleutherAI/pythia-12b-deduped</th>
      <td id="T_fd076_row3_col0" class="data row3 col0" >0.920000</td>
    </tr>
    <tr>
      <th id="T_fd076_level1_row4" class="row_heading level1 row4" >KoboldAI/fairseq-dense-13B</th>
      <td id="T_fd076_row4_col0" class="data row4 col0" >0.960000</td>
    </tr>
    <tr>
      <th id="T_fd076_level1_row5" class="row_heading level1 row5" >Salesforce/codegen-16B-multi</th>
      <td id="T_fd076_row5_col0" class="data row5 col0" >0.920000</td>
    </tr>
    <tr>
      <th id="T_fd076_level1_row6" class="row_heading level1 row6" >ai-forever/rugpt3large_based_on_gpt2</th>
      <td id="T_fd076_row6_col0" class="data row6 col0" >0.940000</td>
    </tr>
    <tr>
      <th id="T_fd076_level1_row7" class="row_heading level1 row7" >bigscience/bloom</th>
      <td id="T_fd076_row7_col0" class="data row7 col0" >0.980000</td>
    </tr>
    <tr>
      <th id="T_fd076_level1_row8" class="row_heading level1 row8" >facebook/bart-large-mnli</th>
      <td id="T_fd076_row8_col0" class="data row8 col0" >0.960000</td>
    </tr>
    <tr>
      <th id="T_fd076_level1_row9" class="row_heading level1 row9" >facebook/galactica-120b</th>
      <td id="T_fd076_row9_col0" class="data row9 col0" >0.940000</td>
    </tr>
    <tr>
      <th id="T_fd076_level1_row10" class="row_heading level1 row10" >facebook/opt-66b</th>
      <td id="T_fd076_row10_col0" class="data row10 col0" >0.960000</td>
    </tr>
    <tr>
      <th id="T_fd076_level1_row11" class="row_heading level1 row11" >gpt2</th>
      <td id="T_fd076_row11_col0" class="data row11 col0" >0.960000</td>
    </tr>
    <tr>
      <th id="T_fd076_level1_row12" class="row_heading level1 row12" >laion/CLIP-ViT-bigG-14-laion2B-39B-b160k</th>
      <td id="T_fd076_row12_col0" class="data row12 col0" >0.600000</td>
    </tr>
    <tr>
      <th id="T_fd076_level1_row13" class="row_heading level1 row13" >microsoft/deberta-base</th>
      <td id="T_fd076_row13_col0" class="data row13 col0" >0.960000</td>
    </tr>
    <tr>
      <th id="T_fd076_level1_row14" class="row_heading level1 row14" >roberta-base</th>
      <td id="T_fd076_row14_col0" class="data row14 col0" >0.960000</td>
    </tr>
    <tr>
      <th id="T_fd076_level1_row15" class="row_heading level1 row15" >sentence-transformers/all-roberta-large-v1</th>
      <td id="T_fd076_row15_col0" class="data row15 col0" >0.960000</td>
    </tr>
    <tr>
      <th id="T_fd076_level1_row16" class="row_heading level1 row16" >stabilityai/stablecode-completion-alpha-3b-4k</th>
      <td id="T_fd076_row16_col0" class="data row16 col0" >0.940000</td>
    </tr>
    <tr>
      <th id="T_fd076_level0_row17" class="row_heading level0 row17" rowspan="16">sentencepiece</th>
      <th id="T_fd076_level1_row17" class="row_heading level1 row17" >NousResearch/Llama-2-13b-hf</th>
      <td id="T_fd076_row17_col0" class="data row17 col0" >1.000000</td>
    </tr>
    <tr>
      <th id="T_fd076_level1_row18" class="row_heading level1 row18" >NousResearch/Llama-2-13b-hf_slow</th>
      <td id="T_fd076_row18_col0" class="data row18 col0" >1.000000</td>
    </tr>
    <tr>
      <th id="T_fd076_level1_row19" class="row_heading level1 row19" >THUDM/chatglm2-6b</th>
      <td id="T_fd076_row19_col0" class="data row19 col0" >0.500000</td>
    </tr>
    <tr>
      <th id="T_fd076_level1_row20" class="row_heading level1 row20" >THUDM/chatglm2-6b_slow</th>
      <td id="T_fd076_row20_col0" class="data row20 col0" >0.500000</td>
    </tr>
    <tr>
      <th id="T_fd076_level1_row21" class="row_heading level1 row21" >THUDM/chatglm3-6b</th>
      <td id="T_fd076_row21_col0" class="data row21 col0" >1.000000</td>
    </tr>
    <tr>
      <th id="T_fd076_level1_row22" class="row_heading level1 row22" >THUDM/chatglm3-6b_slow</th>
      <td id="T_fd076_row22_col0" class="data row22 col0" >1.000000</td>
    </tr>
    <tr>
      <th id="T_fd076_level1_row23" class="row_heading level1 row23" >camembert-base</th>
      <td id="T_fd076_row23_col0" class="data row23 col0" >0.260000</td>
    </tr>
    <tr>
      <th id="T_fd076_level1_row24" class="row_heading level1 row24" >camembert-base_slow</th>
      <td id="T_fd076_row24_col0" class="data row24 col0" >0.260000</td>
    </tr>
    <tr>
      <th id="T_fd076_level1_row25" class="row_heading level1 row25" >codellama/CodeLlama-7b-hf</th>
      <td id="T_fd076_row25_col0" class="data row25 col0" >1.000000</td>
    </tr>
    <tr>
      <th id="T_fd076_level1_row26" class="row_heading level1 row26" >codellama/CodeLlama-7b-hf_slow</th>
      <td id="T_fd076_row26_col0" class="data row26 col0" >1.000000</td>
    </tr>
    <tr>
      <th id="T_fd076_level1_row27" class="row_heading level1 row27" >microsoft/deberta-v3-base</th>
      <td id="T_fd076_row27_col0" class="data row27 col0" >0.940000</td>
    </tr>
    <tr>
      <th id="T_fd076_level1_row28" class="row_heading level1 row28" >microsoft/deberta-v3-base_slow</th>
      <td id="T_fd076_row28_col0" class="data row28 col0" >0.980000</td>
    </tr>
    <tr>
      <th id="T_fd076_level1_row29" class="row_heading level1 row29" >xlm-roberta-base</th>
      <td id="T_fd076_row29_col0" class="data row29 col0" >0.000000</td>
    </tr>
    <tr>
      <th id="T_fd076_level1_row30" class="row_heading level1 row30" >xlm-roberta-base_slow</th>
      <td id="T_fd076_row30_col0" class="data row30 col0" >0.000000</td>
    </tr>
    <tr>
      <th id="T_fd076_level1_row31" class="row_heading level1 row31" >xlnet-base-cased</th>
      <td id="T_fd076_row31_col0" class="data row31 col0" >0.240000</td>
    </tr>
    <tr>
      <th id="T_fd076_level1_row32" class="row_heading level1 row32" >xlnet-base-cased_slow</th>
      <td id="T_fd076_row32_col0" class="data row32 col0" >0.240000</td>
    </tr>
    <tr>
      <th id="T_fd076_level0_row33" class="row_heading level0 row33" rowspan="2">tiktoken</th>
      <th id="T_fd076_level1_row33" class="row_heading level1 row33" >Qwen/Qwen-14B-Chat</th>
      <td id="T_fd076_row33_col0" class="data row33 col0" >0.960000</td>
    </tr>
    <tr>
      <th id="T_fd076_level1_row34" class="row_heading level1 row34" >Salesforce/xgen-7b-8k-base</th>
      <td id="T_fd076_row34_col0" class="data row34 col0" >0.940000</td>
    </tr>
    <tr>
      <th id="T_fd076_level0_row35" class="row_heading level0 row35" rowspan="13">wordpiece</th>
      <th id="T_fd076_level1_row35" class="row_heading level1 row35" >ProsusAI/finbert</th>
      <td id="T_fd076_row35_col0" class="data row35 col0" >1.000000</td>
    </tr>
    <tr>
      <th id="T_fd076_level1_row36" class="row_heading level1 row36" >bert-base-multilingual-cased</th>
      <td id="T_fd076_row36_col0" class="data row36 col0" >1.000000</td>
    </tr>
    <tr>
      <th id="T_fd076_level1_row37" class="row_heading level1 row37" >bert-large-cased</th>
      <td id="T_fd076_row37_col0" class="data row37 col0" >1.000000</td>
    </tr>
    <tr>
      <th id="T_fd076_level1_row38" class="row_heading level1 row38" >cointegrated/rubert-tiny2</th>
      <td id="T_fd076_row38_col0" class="data row38 col0" >1.000000</td>
    </tr>
    <tr>
      <th id="T_fd076_level1_row39" class="row_heading level1 row39" >distilbert-base-uncased-finetuned-sst-2-english</th>
      <td id="T_fd076_row39_col0" class="data row39 col0" >1.000000</td>
    </tr>
    <tr>
      <th id="T_fd076_level1_row40" class="row_heading level1 row40" >google/electra-base-discriminator</th>
      <td id="T_fd076_row40_col0" class="data row40 col0" >1.000000</td>
    </tr>
    <tr>
      <th id="T_fd076_level1_row41" class="row_heading level1 row41" >google/mobilebert-uncased</th>
      <td id="T_fd076_row41_col0" class="data row41 col0" >1.000000</td>
    </tr>
    <tr>
      <th id="T_fd076_level1_row42" class="row_heading level1 row42" >jhgan/ko-sbert-sts</th>
      <td id="T_fd076_row42_col0" class="data row42 col0" >1.000000</td>
    </tr>
    <tr>
      <th id="T_fd076_level1_row43" class="row_heading level1 row43" >prajjwal1/bert-mini</th>
      <td id="T_fd076_row43_col0" class="data row43 col0" >1.000000</td>
    </tr>
    <tr>
      <th id="T_fd076_level1_row44" class="row_heading level1 row44" >rajiv003/ernie-finetuned-qqp</th>
      <td id="T_fd076_row44_col0" class="data row44 col0" >1.000000</td>
    </tr>
    <tr>
      <th id="T_fd076_level1_row45" class="row_heading level1 row45" >rasa/LaBSE</th>
      <td id="T_fd076_row45_col0" class="data row45 col0" >0.896552</td>
    </tr>
    <tr>
      <th id="T_fd076_level1_row46" class="row_heading level1 row46" >sentence-transformers/all-MiniLM-L6-v2</th>
      <td id="T_fd076_row46_col0" class="data row46 col0" >1.000000</td>
    </tr>
    <tr>
      <th id="T_fd076_level1_row47" class="row_heading level1 row47" >squeezebert/squeezebert-uncased</th>
      <td id="T_fd076_row47_col0" class="data row47 col0" >1.000000</td>
    </tr>
  </tbody>
</table>

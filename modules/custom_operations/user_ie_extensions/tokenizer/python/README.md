# OpenVINO Tokenizers

OpenVINO Tokenizers adds text processing operations to OpenVINO.

## Features

- Perform tokenization and detokenization without third-party dependencies
- Convert a HuggingFace tokenizer into OpenVINO model tokenizer and detokenizer
- Combine OpenVINO models into a single model
- Add greedy decoding pipeline to text generation model

## Installation

(Recommended) Create and activate virtual env:
```bash
python3 -m venv venv
source venv/bin/activate
 # or
conda create --name openvino_tokenizer 
conda activate openvino_tokenizer
```

### Minimal Installation

Use minimal installation when you have a converted OpenVINO tokenizer:
```bash
pip install openvino-tokenizers
 # or
conda install -c conda-forge openvino openvino-tokenizers
```

### Convert Tokenizers Installation

If you want to convert HuggingFace tokenizers into OpenVINO tokenizers:
```bash
pip install openvino-tokenizers[transformers]
 # or
conda install -c conda-forge openvino openvino-tokenizers && pip install transformers[sentencepiece] tiktoken
```

### Build and install from source after [OpenVINO installation](https://docs.openvino.ai/2023.2/openvino_docs_install_guides_overview.html)
```bash
source path/to/installed/openvino/setupvars.sh
git clone https://github.com/openvinotoolkit/openvino_contrib.git
cd openvino_contrib/modules/custom_operations/
pip install .[transformers]
```

### Build and install for development
```bash
source path/to/installed/openvino/setupvars.sh
git clone https://github.com/openvinotoolkit/openvino_contrib.git
cd openvino_contrib/modules/custom_operations/
pip install -e .[all]
# verify installation by running tests
cd user_ie_extensions/tokenizer/python/tests/
pytest .
```

### C++ Installation

You can use converted tokenizers in C++ pipelines with prebuild binaries.

1. Download OpenVINO archive distribution for your OS from [here](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/download.html) and extract the archive.
2. Download OpenVINO Tokenizers prebuild libraries from [here](https://storage.openvinotoolkit.org/repositories/openvino_tokenizers/packages/). To ensure compatibility first three numbers of OpenVINO Tokenizers version should match OpenVINO version and OS. 
3. Extract OpenVINO Tokenizers archive into OpenVINO installation directory:
    - Windows: `<openvino_dir>\runtime\bin\intel64\Release\`
    - MacOS_x86: `<openvino_dir>/runtime/lib/intel64/Release`
    - MacOS_arm64: `<openvino_dir>/runtime/lib/arm64/Release/`
    - Linux_x86: `<openvino_dir>/runtime/lib/intel64/`
    - Linux_arm64: `<openvino_dir>/runtime/lib/aarch64/`

After that you can add binary extension in the code with:
- `core.add_extension("openvino_tokenizers.dll")` for Windows
- `core.add_extension("libopenvino_tokenizers.dylib")` for MacOS
- `core.add_extension("libopenvino_tokenizers.so")` for Linux

and `read`/`compile` converted (de)tokenizers models.

## Usage

:warning: OpenVINO Tokenizers can be inferred on a `CPU` device only.

### Convert HuggingFace tokenizer

OpenVINO Tokenizers ships with CLI tool that can convert tokenizers from Huggingface Hub 
or Huggingface tokenizers saved on disk:

```shell
convert_tokenizer codellama/CodeLlama-7b-hf --with-detokenizer -o output_dir
```

There is also `convert_tokenizer` function that can convert tokenizer python object.

```python
import numpy as np
from transformers import AutoTokenizer
from openvino import compile_model, save_model
from openvino_tokenizers import convert_tokenizer

hf_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
ov_tokenizer = convert_tokenizer(hf_tokenizer)

compiled_tokenzier = compile_model(ov_tokenizer)
text_input = ["Test string"]

hf_output = hf_tokenizer(text_input, return_tensors="np")
ov_output = compiled_tokenzier(text_input)

for output_name in hf_output:
    print(f"OpenVINO {output_name} = {ov_output[output_name]}")
    print(f"HuggingFace {output_name} = {hf_output[output_name]}")
# OpenVINO input_ids = [[ 101 3231 5164  102]]
# HuggingFace input_ids = [[ 101 3231 5164  102]]
# OpenVINO token_type_ids = [[0 0 0 0]]
# HuggingFace token_type_ids = [[0 0 0 0]]
# OpenVINO attention_mask = [[1 1 1 1]]
# HuggingFace attention_mask = [[1 1 1 1]]

# save tokenizer for later use
save_model(ov_tokenizer, "openvino_tokenizer.xml")

loaded_tokenizer = compile_model("openvino_tokenizer.xml")
loaded_ov_output = loaded_tokenizer(text_input)
for output_name in hf_output:
    assert np.all(loaded_ov_output[output_name] == ov_output[output_name])
```

### Connect Tokenizer to a Model

To infer and convert the original model, install torch or torch-cpu to the virtual environment.

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from openvino import compile_model, convert_model
from openvino_tokenizers import convert_tokenizer, connect_models

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

openvino_output = compiled_combined_model(text_input)

print(f"OpenVINO logits: {openvino_output['logits']}")
# OpenVINO logits: [[ 1.2007061 -1.4698029]]
print(f"HuggingFace logits {hf_output.logits}")
# HuggingFace logits tensor([[ 1.2007, -1.4698]], grad_fn=<AddmmBackward0>)
```

### Use Extension With Converted (De)Tokenizer or Model With (De)Tokenizer

Import `openvino_tokenizers` will add all tokenizer-related operations to OpenVINO,
after which you can work with saved tokenizers and detokenizers.

```python
import numpy as np
import openvino_tokenizers
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
ov_input = compiled_tokenizer(text_input)
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

## Supported Tokenizer Types

| Huggingface <br/>Tokenizer Type | Tokenizer Model Type | Tokenizer | Detokenizer |
|---------------------------------|----------------------|----------|------------|
| Fast                            | WordPiece            | ✅        | ❌          |
|                                 | BPE                  | ✅        | ✅          |
|                                 | Unigram              | ❌         | ❌          |
| Legacy                          | SentencePiece .model | ✅        | ✅          |
| Custom                          | tiktoken             | ✅        | ✅          |

## Test Results

This report is autogenerated and includes tokenizers and detokenizers tests. The `Output Matched, %` column shows the percent of test strings for which the results of OpenVINO and Hugingface Tokenizers are the same. To update the report run `pytest tokenizers_test.py --update_readme` in `modules/custom_operations/user_ie_extensions/tokenizer/python/tests` directory.

### Output Match by Tokenizer Type

<table>
  <thead>
    <tr>
      <th >Tokenizer Type</th>
      <th >Output Matched, %</th>
      <th >Number of Tests</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td >BPE</td>
      <td >95.82</td>
      <td >3420</td>
    </tr>
    <tr>
      <td >SentencePiece</td>
      <td >86.28</td>
      <td >2880</td>
    </tr>
    <tr>
      <td >Tiktoken</td>
      <td >97.22</td>
      <td >324</td>
    </tr>
    <tr>
      <td >WordPiece</td>
      <td >82.12</td>
      <td >520</td>
    </tr>
  </tbody>
</table>

### Output Match by Model

<table>
  <thead>
    <tr>
      <th >Tokenizer Type</th>
      <th >Model</th>
      <th >Output Matched, %</th>
      <th >Number of Tests</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td >BPE</td>
      <td >EleutherAI/gpt-j-6b</td>
      <td >98.33</td>
      <td >180</td>
    </tr>
    <tr>
      <td >BPE</td>
      <td >EleutherAI/gpt-neo-125m</td>
      <td >98.33</td>
      <td >180</td>
    </tr>
    <tr>
      <td >BPE</td>
      <td >EleutherAI/gpt-neox-20b</td>
      <td >97.78</td>
      <td >180</td>
    </tr>
    <tr>
      <td >BPE</td>
      <td >EleutherAI/pythia-12b-deduped</td>
      <td >97.78</td>
      <td >180</td>
    </tr>
    <tr>
      <td >BPE</td>
      <td >KoboldAI/fairseq-dense-13B</td>
      <td >98.89</td>
      <td >180</td>
    </tr>
    <tr>
      <td >BPE</td>
      <td >Salesforce/codegen-16B-multi</td>
      <td >97.22</td>
      <td >180</td>
    </tr>
    <tr>
      <td >BPE</td>
      <td >ai-forever/rugpt3large_based_on_gpt2</td>
      <td >97.78</td>
      <td >180</td>
    </tr>
    <tr>
      <td >BPE</td>
      <td >bigscience/bloom</td>
      <td >99.44</td>
      <td >180</td>
    </tr>
    <tr>
      <td >BPE</td>
      <td >databricks/dolly-v2-3b</td>
      <td >97.78</td>
      <td >180</td>
    </tr>
    <tr>
      <td >BPE</td>
      <td >facebook/bart-large-mnli</td>
      <td >97.22</td>
      <td >180</td>
    </tr>
    <tr>
      <td >BPE</td>
      <td >facebook/galactica-120b</td>
      <td >98.33</td>
      <td >180</td>
    </tr>
    <tr>
      <td >BPE</td>
      <td >facebook/opt-66b</td>
      <td >98.89</td>
      <td >180</td>
    </tr>
    <tr>
      <td >BPE</td>
      <td >gpt2</td>
      <td >97.22</td>
      <td >180</td>
    </tr>
    <tr>
      <td >BPE</td>
      <td >laion/CLIP-ViT-bigG-14-laion2B-39B-b160k</td>
      <td >61.11</td>
      <td >180</td>
    </tr>
    <tr>
      <td >BPE</td>
      <td >microsoft/deberta-base</td>
      <td >96.11</td>
      <td >180</td>
    </tr>
    <tr>
      <td >BPE</td>
      <td >roberta-base</td>
      <td >96.11</td>
      <td >180</td>
    </tr>
    <tr>
      <td >BPE</td>
      <td >sentence-transformers/all-roberta-large-v1</td>
      <td >96.11</td>
      <td >180</td>
    </tr>
    <tr>
      <td >BPE</td>
      <td >stabilityai/stablecode-completion-alpha-3b-4k</td>
      <td >98.33</td>
      <td >180</td>
    </tr>
    <tr>
      <td >BPE</td>
      <td >stabilityai/stablelm-tuned-alpha-7b</td>
      <td >97.78</td>
      <td >180</td>
    </tr>
    <tr>
      <td >SentencePiece</td>
      <td >NousResearch/Llama-2-13b-hf</td>
      <td >100.00</td>
      <td >180</td>
    </tr>
    <tr>
      <td >SentencePiece</td>
      <td >NousResearch/Llama-2-13b-hf_slow</td>
      <td >100.00</td>
      <td >180</td>
    </tr>
    <tr>
      <td >SentencePiece</td>
      <td >THUDM/chatglm2-6b</td>
      <td >100.00</td>
      <td >180</td>
    </tr>
    <tr>
      <td >SentencePiece</td>
      <td >THUDM/chatglm2-6b_slow</td>
      <td >100.00</td>
      <td >180</td>
    </tr>
    <tr>
      <td >SentencePiece</td>
      <td >THUDM/chatglm3-6b</td>
      <td >100.00</td>
      <td >180</td>
    </tr>
    <tr>
      <td >SentencePiece</td>
      <td >THUDM/chatglm3-6b_slow</td>
      <td >100.00</td>
      <td >180</td>
    </tr>
    <tr>
      <td >SentencePiece</td>
      <td >camembert-base</td>
      <td >0.00</td>
      <td >180</td>
    </tr>
    <tr>
      <td >SentencePiece</td>
      <td >camembert-base_slow</td>
      <td >75.00</td>
      <td >180</td>
    </tr>
    <tr>
      <td >SentencePiece</td>
      <td >codellama/CodeLlama-7b-hf</td>
      <td >100.00</td>
      <td >180</td>
    </tr>
    <tr>
      <td >SentencePiece</td>
      <td >codellama/CodeLlama-7b-hf_slow</td>
      <td >100.00</td>
      <td >180</td>
    </tr>
    <tr>
      <td >SentencePiece</td>
      <td >microsoft/deberta-v3-base</td>
      <td >93.33</td>
      <td >180</td>
    </tr>
    <tr>
      <td >SentencePiece</td>
      <td >microsoft/deberta-v3-base_slow</td>
      <td >100.00</td>
      <td >180</td>
    </tr>
    <tr>
      <td >SentencePiece</td>
      <td >xlm-roberta-base</td>
      <td >98.89</td>
      <td >180</td>
    </tr>
    <tr>
      <td >SentencePiece</td>
      <td >xlm-roberta-base_slow</td>
      <td >98.89</td>
      <td >180</td>
    </tr>
    <tr>
      <td >SentencePiece</td>
      <td >xlnet-base-cased</td>
      <td >61.11</td>
      <td >180</td>
    </tr>
    <tr>
      <td >SentencePiece</td>
      <td >xlnet-base-cased_slow</td>
      <td >53.33</td>
      <td >180</td>
    </tr>
    <tr>
      <td >Tiktoken</td>
      <td >Qwen/Qwen-14B-Chat</td>
      <td >98.15</td>
      <td >108</td>
    </tr>
    <tr>
      <td >Tiktoken</td>
      <td >Salesforce/xgen-7b-8k-base</td>
      <td >97.22</td>
      <td >108</td>
    </tr>
    <tr>
      <td >Tiktoken</td>
      <td >stabilityai/stablelm-2-1_6b</td>
      <td >96.30</td>
      <td >108</td>
    </tr>
    <tr>
      <td >WordPiece</td>
      <td >ProsusAI/finbert</td>
      <td >80.00</td>
      <td >40</td>
    </tr>
    <tr>
      <td >WordPiece</td>
      <td >bert-base-multilingual-cased</td>
      <td >80.00</td>
      <td >40</td>
    </tr>
    <tr>
      <td >WordPiece</td>
      <td >bert-large-cased</td>
      <td >80.00</td>
      <td >40</td>
    </tr>
    <tr>
      <td >WordPiece</td>
      <td >cointegrated/rubert-tiny2</td>
      <td >80.00</td>
      <td >40</td>
    </tr>
    <tr>
      <td >WordPiece</td>
      <td >distilbert-base-uncased-finetuned-sst-2-english</td>
      <td >80.00</td>
      <td >40</td>
    </tr>
    <tr>
      <td >WordPiece</td>
      <td >google/electra-base-discriminator</td>
      <td >80.00</td>
      <td >40</td>
    </tr>
    <tr>
      <td >WordPiece</td>
      <td >google/mobilebert-uncased</td>
      <td >95.00</td>
      <td >40</td>
    </tr>
    <tr>
      <td >WordPiece</td>
      <td >jhgan/ko-sbert-sts</td>
      <td >75.00</td>
      <td >40</td>
    </tr>
    <tr>
      <td >WordPiece</td>
      <td >prajjwal1/bert-mini</td>
      <td >95.00</td>
      <td >40</td>
    </tr>
    <tr>
      <td >WordPiece</td>
      <td >rajiv003/ernie-finetuned-qqp</td>
      <td >95.00</td>
      <td >40</td>
    </tr>
    <tr>
      <td >WordPiece</td>
      <td >rasa/LaBSE</td>
      <td >72.50</td>
      <td >40</td>
    </tr>
    <tr>
      <td >WordPiece</td>
      <td >sentence-transformers/all-MiniLM-L6-v2</td>
      <td >75.00</td>
      <td >40</td>
    </tr>
    <tr>
      <td >WordPiece</td>
      <td >squeezebert/squeezebert-uncased</td>
      <td >80.00</td>
      <td >40</td>
    </tr>
  </tbody>
</table>

### Recreating Tokenizers From Tests

In some tokenizers, you need to select certain settings so that their output is closer to the Huggingface tokenizers:
- `THUDM/chatglm2-6b` detokenizer always skips special tokens. Use `skip_special_tokens=True` during conversion
- `THUDM/chatglm3-6b` detokenizer don't skips special tokens. Use `skip_special_tokens=False` during conversion
- All tested tiktoken based detokenizers leave extra spaces. Use `clean_up_tokenization_spaces=False` during conversion

# Use OpenVINO for Masked Language Modeling
#
# This demo shows how to use openvino-optimum with a PyTorch Fill-Mask model
# from https://huggingface.co/models
#
# The code demonstrates how to load a model from the Hugging Face Model Hub
# and do inference with OpenVINO. Model conversion and inference with
# OpenVINO is handled by openvino-optimum in the background.
#
# To run this model, please install openvino-optimum[all]
# (see https://github.com/openvinotoolkit/openvino_contrib/blob/master/modules/optimum/README.md )
# and PyTorch. The demo is tested with torch==1.13.1.
#
# NOTE: openvino-optimum initializes the model on first inference, so the first inference
#       is expected to take longer.

import torch
from optimum.intel.openvino import OVAutoModelForMaskedLM
from transformers import AutoTokenizer

# Please refer to https://huggingface.co/roberta-base for more information
# about this model, including the Limitations and Bias section:
# https://huggingface.co/roberta-base#limitations-and-bias
model_name = "roberta-base"

# Also try one of the following models:
#
# "bert-base-uncased", "roberta-base", "albert-base-v2", "albert-large-v2", "distilbert-base-uncased"
# "xlm-roberta-base" # "bert-large-uncased", "bert-base-multilingual-cased", "bert-base-multilingual-uncased"
# "roberta-large", "distilroberta-base"

# Set num_sentences to the number of sentences to predict
num_sentences = 5


# Load OpenVINO model from PyTorch
# If the model has not been downloaded before, it will be downloaded from
# the Hugging Face Model Hub. openvino-optimum will convert the model to ONNX
# in the background.
model = OVAutoModelForMaskedLM.from_pretrained(model_name, return_dict=True, from_pt=True)

# To compare the results of the OpenVINO model with the Hugging Face PyTorch model,
# replace the OVAutoModel line with AutoModel, as in the two commented-out lines below
#
# from transformers import AutoModelForMaskedLM, AutoTokenizer
# model = AutoModelForMaskedLM.from_pretrained(model_name, return_dict=True)

# Optional: save the pretrained model to disk
# model.save_pretrained(f"{model_name}_openvino")

tokenizer = AutoTokenizer.from_pretrained(model_name)

# Demo text from https://huggingface.co/roberta-base
text = f"The goal of life is {tokenizer.mask_token}."

# Convert text to tokens
encoded_input = tokenizer(text, return_tensors="pt")

# Find position of <mask> token
mask_index = torch.where(encoded_input["input_ids"].squeeze() == tokenizer.mask_token_id)

# Do inference
output = model(**encoded_input)

# Find the top k results
softmax_result = output.logits.softmax(dim=-1)
mask_word = softmax_result[0, mask_index, :]
top_tokens = torch.topk(mask_word, num_sentences, dim=1)[1][0]

for tokens in top_tokens:
    # Convert tokens to word
    word = tokenizer.decode([tokens], skip_special_tokens=False)
    new_sentence = text.replace(tokenizer.mask_token, word)
    print(new_sentence)

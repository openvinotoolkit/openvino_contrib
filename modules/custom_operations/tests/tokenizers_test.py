import sys

sys.path.append("../user_ie_extensions/tokenizer")
# import os
# os.environ["OV_TOKENIZER_PREBUILD_EXTENSION_PATH"] = "path/to/libuser_ov_extensions.so"

import pytest
import numpy as np
from openvino import Core
from transformers import AutoTokenizer
from ov_tokenizer import (
    # init_extension,
    convert_tokenizer,
    connect_models,
    pack_strings,
    unpack_strings,
    convert_sentencepiece_model_tokenizer
)


# use `init_extension` function to be able to convert HF tokenizers:
# init_extension("path/to/libuser_ov_extensions.so")  # or alternatively:
# set the OV_TOKENIZER_PREBUILD_EXTENSION_PATH env variable BEFORE importing ov_tokenizers
core = Core()

eng_test_strings = [
    "Eng... test, string?!",
    "Multiline\nstring!\nWow!",
    "A lot\t w!",
    "A lot\t\tof whitespaces!",
    "\n\n\n\t\t   A    lot\t\tof\twhitespaces\n!\n\n\n\t\n\n",
    "Eng, but with d1gits: 123; 0987654321, stop."
    "0987654321 - eng, but with d1gits: 123"
]
multilingual_test_strings = [
    "Тестовая строка!",
    "Testzeichenfolge?",
    "Tester, la chaîne...",
    "測試字符串",
    "سلسلة الاختبار",
    "מחרוזת בדיקה",
    "Сынақ жолы",
    "رشته تست",
]
emoji_test_strings = [
    "😀",
    "😁😁",
    "🤣🤣🤣😁😁😁😁",
    "🫠",  # melting face
    "🤷‍♂️",
    "🤦🏼‍♂️",
]

wordpiece_models = [
    "bert-base-multilingual-cased",
    "bert-large-cased",
    "cointegrated/rubert-tiny2",
    "distilbert-base-uncased-finetuned-sst-2-english",
    "sentence-transformers/all-MiniLM-L6-v2",
    "rajiv003/ernie-finetuned-qqp",  # ernie model with fast tokenizer
    "google/electra-base-discriminator",
    "google/mobilebert-uncased",
    "jhgan/ko-sbert-sts",
    "squeezebert/squeezebert-uncased",
    "prajjwal1/bert-mini",
    "ProsusAI/finbert",
    "rasa/LaBSE",
]
bpe_models = [
    "stabilityai/stablecode-completion-alpha-3b-4k",
    "EleutherAI/gpt-neo-125m",
    "EleutherAI/gpt-j-6b",
    "roberta-base",
    "sentence-transformers/all-roberta-large-v1",  # standin for setfit
    "facebook/bart-large-mnli",
    "facebook/opt-66b",
    "gpt2",
    "EleutherAI/gpt-neox-20b",
    "ai-forever/rugpt3large_based_on_gpt2",
    "KoboldAI/fairseq-dense-13B",
    "facebook/galactica-120b",
    "EleutherAI/pythia-12b-deduped",
    "Salesforce/codegen-16B-multi",
    "microsoft/deberta-base",
    "bigscience/bloom",  # pack_strings for vocab is taking long time
    "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
    # "google/flan-t5-xxl",  # needs Precompiled/CharsMap
    # "decapoda-research/llama-65b-hf",  # not importable from hub
    # "jinmang2/textcnn-ko-dialect-classifier",  # Needs Metaspace Pretokenizer
    # "hyunwoongko/blenderbot-9B",  # hf script to get fast tokenizer doesn't work
]
sentencepiece_models = [
    "codellama/CodeLlama-7b-hf",
    "camembert-base",
    "NousResearch/Llama-2-13b-hf",
    "xlm-roberta-base",
    "microsoft/deberta-v3-base",
    "xlnet-base-cased",
    # "t5-base",  # crashes tests
]


@pytest.fixture(scope="session", params=wordpiece_models, ids=lambda checkpoint: checkpoint.split("/")[-1])
def hf_and_ov_wordpiece_tokenizers(request):
    hf_tokenizer = AutoTokenizer.from_pretrained(request.param, use_fast=True)
    ov_tokenizer = convert_tokenizer(hf_tokenizer)
    compiled_tokenizer = core.compile_model(ov_tokenizer)
    return hf_tokenizer, compiled_tokenizer


@pytest.fixture(scope="session", params=bpe_models, ids=lambda checkpoint: checkpoint.split("/")[-1])
def hf_and_ov_bpe_tokenizers(request):
    hf_tokenizer = AutoTokenizer.from_pretrained(request.param, use_fast=True)
    ov_tokenizer = convert_tokenizer(hf_tokenizer)
    compiled_tokenizer = core.compile_model(ov_tokenizer)
    return hf_tokenizer, compiled_tokenizer


@pytest.fixture(scope="session", params=[True, False], ids=lambda is_fast: "Fast" if is_fast else "Slow")
def fast_tokenzier(request):
    return request.param


@pytest.fixture(scope="session", params=sentencepiece_models, ids=lambda checkpoint: checkpoint.split("/")[-1])
def sentencepice_model_tokenizers(request, fast_tokenzier):
    hf_tokenizer = AutoTokenizer.from_pretrained(request.param, use_fast=fast_tokenzier)
    ov_tokenizer = convert_sentencepiece_model_tokenizer(hf_tokenizer)
    compiled_tokenizer = core.compile_model(ov_tokenizer)
    return hf_tokenizer, compiled_tokenizer


@pytest.mark.parametrize(
    "test_string",
    [
        *eng_test_strings,
        *multilingual_test_strings,
        *emoji_test_strings,
    ]
)
def test_hf_wordpiece_tokenizers_outputs(hf_and_ov_wordpiece_tokenizers, test_string):
    hf_tokenizer, ov_tokenizer = hf_and_ov_wordpiece_tokenizers
    packed_strings = pack_strings([test_string])

    hf_tokenized = hf_tokenizer([test_string], return_tensors="np")
    ov_tokenized = ov_tokenizer(packed_strings)

    for output_name, hf_result in hf_tokenized.items():
        assert np.all((ov_result := ov_tokenized[output_name]) == hf_result), f"{hf_result}\n{ov_result}"


@pytest.mark.parametrize(
    "test_string",
    [
        eng_test_strings,
        multilingual_test_strings,
        emoji_test_strings,
    ]
)
def test_hf_wordpiece_tokenizers_multiple_strings(hf_and_ov_wordpiece_tokenizers, test_string):
    hf_tokenizer, ov_tokenizer = hf_and_ov_wordpiece_tokenizers
    packed_strings = pack_strings(test_string)

    hf_tokenized = hf_tokenizer(test_string, return_tensors="np", padding=True)
    ov_tokenized = ov_tokenizer(packed_strings)

    for output_name, hf_result in hf_tokenized.items():
        assert np.all((ov_result := ov_tokenized[output_name]) == hf_result), f"{hf_result}\n{ov_result}"


@pytest.mark.parametrize(
    "test_string",
    [
        *eng_test_strings,
        *multilingual_test_strings,
        *emoji_test_strings,
    ]
)
def test_hf_bpe_tokenizers_outputs(hf_and_ov_bpe_tokenizers, test_string):
    hf_tokenizer, ov_tokenizer = hf_and_ov_bpe_tokenizers
    packed_strings = pack_strings([test_string])

    hf_tokenized = hf_tokenizer([test_string], return_tensors="np")
    ov_tokenized = ov_tokenizer(packed_strings)

    for output_name, hf_result in hf_tokenized.items():
        ov_result = ov_tokenized.get(output_name)
        # galactica tokenizer has 3 output, but model has 2 inputs
        if ov_result is not None:
            assert np.all(ov_result == hf_result), f"{hf_result}\n{ov_result}"


@pytest.mark.parametrize(
    "test_string",
    [
        *eng_test_strings,
        *multilingual_test_strings,
        *emoji_test_strings,
    ]
)
def test_sentencepiece_model_tokenizer(sentencepice_model_tokenizers, test_string):
    hf_tokenizer, ov_tokenizer = sentencepice_model_tokenizers

    hf_tokenized = hf_tokenizer(test_string, return_tensors="np")
    ov_tokenized = ov_tokenizer(pack_strings([test_string]))

    for output_name, hf_result in hf_tokenized.items():
        assert np.all((ov_result := ov_tokenized[output_name]) == hf_result), f"{hf_result}\n{ov_result}"

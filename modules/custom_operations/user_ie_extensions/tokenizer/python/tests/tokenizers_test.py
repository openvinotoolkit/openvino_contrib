# -*- coding: utf-8 -*-
# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


# import os
# os.environ["OV_TOKENIZER_PREBUILD_EXTENSION_PATH"] = "path/to/libuser_ov_extensions.so"

import numpy as np
import pytest
from openvino import Core
from transformers import AutoTokenizer

from ov_tokenizer import (
    convert_tokenizer,
    pack_strings,
    unpack_strings,
)


core = Core()

eng_test_strings = [
    "Eng... test, string?!",
    "Multiline\nstring!\nWow!",
    "A lot\t w!",
    "A lot\t\tof whitespaces!",
    "\n\n\n\t\t   A    lot\t\tof\twhitespaces\n!\n\n\n\t\n\n",
    "Eng, but with d1gits: 123; 0987654321, stop." "0987654321 - eng, but with d1gits: 123",
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
misc_strings = [
    "",
    " ",
    " " * 10,
    "\n",
    " \t\n",
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
    "microsoft/deberta-base",
    "bigscience/bloom",
    "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
    # "Salesforce/codegen-16B-multi",  # Segfalts on ""A lot\t\tof whitespaces!""
    # "google/flan-t5-xxl",  # needs Precompiled/CharsMap
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
    # "THUDM/chatglm-6b",  # hf_tokenizer init error
    "THUDM/chatglm2-6b",  # detokenizer cannot filter special tokens
    "THUDM/chatglm3-6b",
    # "t5-base",  # crashes tests
]
tiktiken_models = [
    "Qwen/Qwen-14B-Chat",
    "Salesforce/xgen-7b-8k-base",
]


def get_tokenizer(request, fast_tokenizer=True, trust_remote_code=False):
    hf_tokenizer = AutoTokenizer.from_pretrained(
        request.param, use_fast=fast_tokenizer, trust_remote_code=trust_remote_code
    )
    ov_tokenizer = convert_tokenizer(hf_tokenizer, with_decoder=False)
    compiled_tokenizer = core.compile_model(ov_tokenizer)
    return hf_tokenizer, compiled_tokenizer


def get_tokenizer_detokenizer(request, fast_tokenizer=True, trust_remote_code=False):
    hf_tokenizer = AutoTokenizer.from_pretrained(
        request.param, use_fast=fast_tokenizer, trust_remote_code=trust_remote_code
    )
    ov_tokenizer, ov_detokenizer = convert_tokenizer(hf_tokenizer, with_decoder=True)
    compiled_tokenizer = core.compile_model(ov_tokenizer)
    compiled_detokenizer = core.compile_model(ov_detokenizer)
    return hf_tokenizer, compiled_tokenizer, compiled_detokenizer


@pytest.fixture(scope="session", params=wordpiece_models, ids=lambda checkpoint: checkpoint.split("/")[-1])
def hf_and_ov_wordpiece_tokenizers(request):
    return get_tokenizer(request)


@pytest.fixture(scope="session", params=bpe_models, ids=lambda checkpoint: checkpoint.split("/")[-1])
def hf_and_ov_bpe_tokenizers(request):
    return get_tokenizer_detokenizer(request)


@pytest.fixture(scope="session", params=bpe_models, ids=lambda checkpoint: checkpoint.split("/")[-1])
def hf_and_ov_bpe_detokenizer(request):
    return get_tokenizer_detokenizer(request)


@pytest.fixture(scope="session", params=[True, False], ids=lambda is_fast: "Fast" if is_fast else "Slow")
def is_fast_tokenizer(request):
    return request.param


@pytest.fixture(scope="session", params=sentencepiece_models, ids=lambda checkpoint: checkpoint.split("/")[-1])
def sentencepice_model_tokenizers(request, is_fast_tokenizer):
    return get_tokenizer_detokenizer(request, is_fast_tokenizer, trust_remote_code=True)


@pytest.fixture(scope="session", params=tiktiken_models, ids=lambda checkpoint: checkpoint.split("/")[-1])
def tiktoken_tokenizers(request):
    return get_tokenizer(request, trust_remote_code=True)


@pytest.fixture(scope="session", params=tiktiken_models, ids=lambda checkpoint: checkpoint.split("/")[-1])
def tiktoken_detokenizers(request):
    return get_tokenizer_detokenizer(request, trust_remote_code=True)


@pytest.mark.parametrize(
    "test_string",
    [
        *eng_test_strings,
        *multilingual_test_strings,
        *emoji_test_strings,
        *misc_strings,
    ],
)
def test_hf_wordpiece_tokenizers(hf_and_ov_wordpiece_tokenizers, test_string):
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
        misc_strings,
    ],
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
        *misc_strings,
    ],
)
def test_sentencepiece_model_tokenizer(sentencepice_model_tokenizers, test_string):
    hf_tokenizer, ov_tokenizer, _ = sentencepice_model_tokenizers

    hf_tokenized = hf_tokenizer(test_string, return_tensors="np")
    ov_tokenized = ov_tokenizer(pack_strings([test_string]))

    for output_name, hf_result in hf_tokenized.items():
        #  chatglm has token_type_ids output that we omit
        if (ov_result := ov_tokenized.get(output_name)) is not None:
            assert np.all(ov_result == hf_result), f"{hf_result}\n{ov_result}"


@pytest.mark.parametrize(
    "test_string",
    [
        *eng_test_strings,
        *multilingual_test_strings,
        *emoji_test_strings,
        *misc_strings,
    ],
)
def test_sentencepiece_model_detokenizer(sentencepice_model_tokenizers, test_string):
    hf_tokenizer, _, ov_detokenizer = sentencepice_model_tokenizers

    token_ids = hf_tokenizer(test_string, return_tensors="np").input_ids
    hf_output = hf_tokenizer.batch_decode(token_ids, skip_special_tokens=True)
    ov_output = unpack_strings(ov_detokenizer(token_ids.astype("int32"))["string_output"])

    assert ov_output == hf_output


@pytest.mark.parametrize(
    "test_string",
    [
        *eng_test_strings,
        *multilingual_test_strings,
        *emoji_test_strings,
        *misc_strings,
    ],
)
def test_hf_bpe_tokenizers_outputs(hf_and_ov_bpe_tokenizers, test_string):
    hf_tokenizer, ov_tokenizer, _ = hf_and_ov_bpe_tokenizers
    packed_strings = pack_strings([test_string])

    hf_tokenized = hf_tokenizer([test_string], return_tensors="np")
    ov_tokenized = ov_tokenizer(packed_strings)

    for output_name, hf_result in hf_tokenized.items():
        # galactica tokenizer has 3 output, but model has 2 inputs
        if (ov_result := ov_tokenized.get(output_name)) is not None:
            assert np.all(ov_result == hf_result), f"{hf_result}\n{ov_result}"


@pytest.mark.parametrize(
    "test_string",
    [
        *eng_test_strings,
        *multilingual_test_strings,
        *emoji_test_strings,
        *misc_strings,
    ],
)
def test_bpe_detokenizer(hf_and_ov_bpe_detokenizer, test_string):
    hf_tokenizer, _, ov_detokenizer = hf_and_ov_bpe_detokenizer

    token_ids = hf_tokenizer(test_string, return_tensors="np").input_ids
    hf_output = hf_tokenizer.batch_decode(token_ids)
    ov_output = unpack_strings(ov_detokenizer(token_ids.astype("int32"))["string_output"])

    assert ov_output == hf_output


# @pytest.mark.skip(reason="tiktoken tokenizer is WIP")
@pytest.mark.parametrize(
    "test_string",
    [
        *eng_test_strings,
        *multilingual_test_strings,
        *emoji_test_strings,
        *misc_strings,
    ],
)
def test_tiktoken_tokenizers(tiktoken_tokenizers, test_string):
    hf_tokenizer, ov_tokenizer = tiktoken_tokenizers

    hf_tokenized = hf_tokenizer(test_string, return_tensors="np")
    ov_tokenized = ov_tokenizer(pack_strings([test_string]))

    for output_name, hf_result in hf_tokenized.items():
        if (ov_result := ov_tokenized.get(output_name)) is not None:
            assert np.all(ov_result == hf_result), f"{hf_result}\n{ov_result}"


@pytest.mark.parametrize(
    "test_string",
    [
        *eng_test_strings,
        *multilingual_test_strings,
        *emoji_test_strings,
        *misc_strings,
    ],
)
def test_tiktoken_detokenizer(tiktoken_detokenizers, test_string):
    hf_tokenizer, _, ov_detokenizer = tiktoken_detokenizers

    token_ids = hf_tokenizer(test_string, return_tensors="np").input_ids
    hf_output = hf_tokenizer.batch_decode(token_ids, skip_special_tokens=True)
    ov_output = unpack_strings(ov_detokenizer(token_ids.astype("int32"))["string_output"])

    assert ov_output == hf_output

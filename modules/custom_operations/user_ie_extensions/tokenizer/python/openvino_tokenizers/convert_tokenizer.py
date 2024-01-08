# -*- coding: utf-8 -*-
# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import sys
from typing import Any, Optional, Tuple, Union

from openvino.runtime import Model, Type
from openvino.runtime.exceptions import OVTypeError

from .utils import change_inputs_type, change_outputs_type


logger = logging.getLogger(__name__)


def convert_tokenizer(
    tokenizer_object: Any,
    with_detokenizer: bool = False,
    skip_special_tokens: bool = False,
    clean_up_tokenization_spaces: Optional[bool] = None,
    tokenizer_output_type: Type = Type.i64,
    detokenizer_input_type: Type = Type.i64,
    streaming_detokenizer: bool = False,
) -> Union[Model, Tuple[Model, Model]]:
    ov_tokenizers = None

    if "transformers" in sys.modules:
        from transformers import PreTrainedTokenizerBase, PreTrainedTokenizerFast

        from .hf_parser import (
            convert_fast_tokenizer,
            convert_sentencepiece_model_tokenizer,
            convert_tiktoken_model_tokenizer,
            is_sentencepiece_model,
            is_tiktoken_model,
        )

        if isinstance(tokenizer_object, PreTrainedTokenizerBase):
            if is_sentencepiece_model(tokenizer_object):
                logger.info("Convert tokenizer using SentencePiece .model file.")
                ov_tokenizers = convert_sentencepiece_model_tokenizer(
                    tokenizer_object,
                    add_attention_mask=True,
                    with_detokenizer=with_detokenizer,
                    streaming_detokenizer=streaming_detokenizer,
                    skip_special_tokens=skip_special_tokens,
                    clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                )
            elif is_tiktoken_model(tokenizer_object):
                logger.info("Convert tiktoken-based tokenizer")
                ov_tokenizers = convert_tiktoken_model_tokenizer(
                    tokenizer_object,
                    with_detokenizer=with_detokenizer,
                    skip_special_tokens=skip_special_tokens,
                    clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                )
            elif isinstance(tokenizer_object, PreTrainedTokenizerFast):
                logger.info("Convert Huggingface Fast tokenizer pipeline.")
                ov_tokenizers = convert_fast_tokenizer(
                    tokenizer_object,
                    number_of_inputs=1,
                    with_detokenizer=with_detokenizer,
                    skip_special_tokens=skip_special_tokens,
                    clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                )
    else:
        raise EnvironmentError(
            "No transformers library in the environment. Install required dependencies with one of two options:\n"
            "1. pip install openvino-tokenizers[transformers]\n"
            "2. pip install transformers[sentencepiece] tiktoken\n"
        )

    if ov_tokenizers is None:
        raise OVTypeError(f"Tokenizer type is not supported: {type(tokenizer_object)}")

    if isinstance(ov_tokenizers, tuple):
        return (
            change_outputs_type(ov_tokenizers[0], tokenizer_output_type),
            change_inputs_type(ov_tokenizers[1], detokenizer_input_type),
        )

    return change_outputs_type(ov_tokenizers, tokenizer_output_type)

# -*- coding: utf-8 -*-
# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import sys
from typing import Any, Tuple, Union

from openvino.runtime import Model, Type
from openvino.runtime.exceptions import OVTypeError

from .utils import change_outputs_type


logger = logging.getLogger(__name__)


def convert_tokenizer(
    tokenizer_object: Any,
    number_of_inputs: int = 1,
    with_decoder: bool = False,
    streaming_decoder: bool = False,
    tokenizer_output_type: Type = Type.i64,
) -> Union[Model, Tuple[Model, Model]]:
    # todo: add support for more then 1 input
    if number_of_inputs > 1:
        raise ValueError("Tokenizers with more then one input are not supported yet.")

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
                    with_decoder=with_decoder,
                    streaming_decoder=streaming_decoder,
                )
            elif is_tiktoken_model(tokenizer_object):
                logger.info("Convert tiktoken-based tokenizer")
                ov_tokenizers = convert_tiktoken_model_tokenizer(
                    tokenizer_object,
                    add_attention_mask=True,
                    with_decoder=with_decoder,
                    streaming_decoder=streaming_decoder,
                )
            elif isinstance(tokenizer_object, PreTrainedTokenizerFast):
                logger.info("Convert Huggingface Fast tokenizer pipeline.")
                ov_tokenizers = convert_fast_tokenizer(
                    tokenizer_object,
                    number_of_inputs=number_of_inputs,
                    with_decoder=with_decoder,
                )

    if ov_tokenizers is None:
        raise OVTypeError(f"Tokenizer type is not supported: {type(tokenizer_object)}")

    if tokenizer_output_type == Type.i32:
        return ov_tokenizers

    if isinstance(ov_tokenizers, tuple):
        return change_outputs_type(ov_tokenizers[0], tokenizer_output_type), ov_tokenizers[1]

    return change_outputs_type(ov_tokenizers, tokenizer_output_type)

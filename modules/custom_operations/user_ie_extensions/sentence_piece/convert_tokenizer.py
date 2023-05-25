# -*- coding: utf-8 -*-
# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sys
from typing import Any, List

from openvino.runtime.exceptions import OVTypeError
from openvino.runtime import Model, Output
from tokenizer_pipeline import TokenizerPipeline


def convert_tokenizer(tokenizer_object: Any, number_of_inputs: int = 1) -> TokenizerPipeline:
    if "transformers" in sys.modules:
        from transformers import PreTrainedTokenizerBase
        from hf_parser import TransformersTokenizerPipelineParser

        if isinstance(tokenizer_object, PreTrainedTokenizerBase):
            return TransformersTokenizerPipelineParser(tokenizer_object).parse(number_of_inputs=number_of_inputs)


    raise OVTypeError(f"Tokenizer type is not supported: {type(tokenizer_object)}")


def connect_tokenizer(model: Model, tokenizer: Model) -> Model:
    assert len(model.inputs) == len(tokenizer.outputs)

    # need to check if the inputs are aligned:
    # - inputs_ids -> inputs_ids
    # - attention_mask -> attention_mask
    # - token_type_ids -> token_type_ids
    aligned_model_inputs = model.inputs
    aligned_tokenizer_outputs: List[Output] = tokenizer.outputs

    for model_input, tokenizer_output in zip(aligned_model_inputs, aligned_tokenizer_outputs):
        for target in model_input.get_target_inputs():
            target.replace_source_output(tokenizer_output)

    return Model(model.outputs, tokenizer.inputs)

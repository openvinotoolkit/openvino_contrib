# -*- coding: utf-8 -*-
# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sys
from typing import Any, List

from openvino.runtime.exceptions import OVTypeError
from openvino.runtime import Model
from tokenizer_pipeline import TokenizerPipeline


def convert_tokenizer(tokenizer_object: Any, number_of_inputs: int = 1) -> TokenizerPipeline:
    if "transformers" in sys.modules:
        from transformers import PreTrainedTokenizerBase
        from hf_parser import TransformersTokenizerPipelineParser

        # TODO: Remove this check
        if isinstance(tokenizer_object, PreTrainedTokenizerBase):
            ov_tokenizer = TransformersTokenizerPipelineParser(tokenizer_object).parse(number_of_inputs=number_of_inputs).get_ov_subgraph()
            output_names = tokenizer_object.model_input_names
            for i, output_name in enumerate(output_names):
                ov_tokenizer.output(i).tensor.add_names({output_name})
            return ov_tokenizer

    raise OVTypeError(f"Tokenizer type is not supported: {type(tokenizer_object)}")


def connect_models(model1: Model, model2: Model, name_map=None, *, by_indices=None, by_names=None) -> Model:
    # TODO: Relax this limitation by not connecting some inputs/outputs together
    assert len(model2.inputs) == len(model1.outputs)

    if by_indices is None and by_names is None:
        by_names = True

    if name_map is not None:
        by_names = True

    # TODO: Check only one of by_indices and by_names is set

    if by_indices:
        aligned_model1_outputs = model1.outputs
        aligned_model2_inputs = model2.inputs
    elif by_names:
        if name_map is None:
            aligned_model1_outputs = model1.outputs
            aligned_model2_inputs = [model2.input(model1_output.get_any_name()) for model1_output in aligned_model1_outputs]
        else:
            aligned_model1_outputs = [model1.output(name1) for name1, _ in name_map]
            aligned_model2_inputs = [model2.input(name2) for _, name2 in name_map]

    for model2_input, model1_output in zip(aligned_model2_inputs, aligned_model1_outputs):
        print(f'Connecting: {model1_output.get_any_name()} -> {model2_input.get_any_name()}')
        for target in model2_input.get_target_inputs():
            target.replace_source_output(model1_output.get_node().input_value(0))
            #target.replace_source_output(model1_output)  # TODO: Produces incorrect topology

    connected_model = Model(model2.outputs, model1.get_parameters())
    # TODO: Cleanup model1 and mode2 to avoid using them, they are ill-formed after the reconnection
    connected_model.validate_nodes_and_infer_types()
    return connected_model

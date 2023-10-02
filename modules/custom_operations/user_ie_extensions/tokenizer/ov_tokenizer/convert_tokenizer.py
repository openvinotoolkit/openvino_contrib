# -*- coding: utf-8 -*-
# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sys
import logging
from typing import Any, Tuple, Union, Optional, Sequence

from openvino.runtime.exceptions import OVTypeError
from openvino.runtime import Model


logger = logging.getLogger(__name__)


def convert_tokenizer(
    tokenizer_object: Any, number_of_inputs: int = 1, with_decoder: bool = False
) -> Union[Model, Tuple[Model, Model]]:
    if "transformers" in sys.modules:
        from transformers import PreTrainedTokenizerBase
        from .hf_parser import TransformersTokenizerPipelineParser

        # TODO: Remove this check
        if isinstance(tokenizer_object, PreTrainedTokenizerBase):
            pipeline = TransformersTokenizerPipelineParser(tokenizer_object).parse(number_of_inputs=number_of_inputs)
            ov_tokenizer = pipeline.get_encoder_ov_subgraph()
            if with_decoder:
                ov_detokenizer = pipeline.get_decoder_ov_subgraph()
            output_names = tokenizer_object.model_input_names

            ov_tokenizer_output_names = ["input_ids", "attention_mask"]
            if len(output_names) == 3 and len(ov_tokenizer.outputs) == 3:
                ov_tokenizer_output_names.insert(1, "token_type_ids")

            filtered_outputs = []
            for i, output_name in enumerate(ov_tokenizer_output_names):
                current_output = next(
                    (output for output in ov_tokenizer.outputs if output.any_name == output_name),
                    False,
                )
                if current_output:
                    filtered_outputs.append(current_output)
                    continue

                if output_name in output_names:
                    ov_tokenizer.output(i).tensor.add_names({output_name})
                    filtered_outputs.append(ov_tokenizer.output(i))

            if with_decoder:
                return (
                    Model(filtered_outputs, ov_tokenizer.get_parameters()),
                    ov_detokenizer,
                )

            return Model(filtered_outputs, ov_tokenizer.get_parameters())

    raise OVTypeError(f"Tokenizer type is not supported: {type(tokenizer_object)}")


def connect_models(
    first: Model,
    second: Model,
    name_map: Optional[Sequence[Tuple[str, str]]] = None,
    by_indices: bool = False,
    keep_unaligned_inputs: bool = True,
    keep_unaligned_outputs: bool = False,
) -> Model:
    if by_indices:
        min_len = min(len(first.outputs), len(second.inputs))
        aligned_first_outputs = first.outputs[:min_len]
        aligned_second_inputs = second.inputs[:min_len]
    elif name_map is None:
        aligned_first_outputs = first.outputs
        aligned_second_inputs = [second.input(model1_output.get_any_name()) for model1_output in aligned_first_outputs]
    else:
        aligned_first_outputs = [first.output(name1) for name1, _ in name_map]
        aligned_second_inputs = [second.input(name2) for _, name2 in name_map]

    for second_input, first_output in zip(aligned_second_inputs, aligned_first_outputs):
        logger.debug(f"Connecting: {first_output.get_any_name()} -> {second_input.get_any_name()}")
        for target in second_input.get_target_inputs():
            target.replace_source_output(first_output.get_node().input_value(0))
            # target.replace_source_output(model1_output)  # TODO: Produces incorrect topology

    new_inputs = first.get_parameters()
    remaining_inputs = [input_ for input_ in second.inputs if input_ not in aligned_second_inputs]
    if keep_unaligned_inputs:
        new_inputs.extend(remaining_inputs)
    elif remaining_inputs:
        logger.info(
            "Some inputs of the second model were left uncovered and not included in the connected model: "
            + ", ".join(input_.name for input_ in remaining_inputs)
            + ". To add them set `keep_unaligned_inputs` to `True`"
        )

    new_outputs = second.outputs
    remaining_outputs = [output for output in first.outputs if output not in aligned_first_outputs]
    if keep_unaligned_outputs:
        new_outputs.extend(remaining_outputs)
    elif remaining_outputs:
        logger.info(
            "Some outputs of the first model were left uncovered and not included in the connected model: "
            + ", ".join(output.name for output in remaining_outputs)
            + ". To add them set `keep_unaligned_outputs` to `True`"
        )

    connected_model = Model(new_outputs, new_inputs, f"{first.get_name()}_{second.get_name()}")
    # TODO: Cleanup model1 and mode2 to avoid using them, they are ill-formed after the reconnection
    connected_model.validate_nodes_and_infer_types()
    return connected_model

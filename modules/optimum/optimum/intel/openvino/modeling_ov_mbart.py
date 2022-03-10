# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
from transformers import MBartForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput

from .modeling_ov_utils import (
    OVPreTrainedModel,
    load_ov_model_from_pytorch,
)

class OVMBartEncoder(OVPreTrainedModel):
    def __init__(self, net, config):
        super().__init__(net, config)

    def __call__(self, *args, **kwargs):
        kwargs["return_dict"] = False
        res = super().__call__(*args, **kwargs)
        return BaseModelOutput(last_hidden_state=torch.tensor(res[0]))


class OVMBartForConditionalGeneration(object):
    @classmethod
    def from_pretrained(cls, model_name_or_path, *model_args, **kwargs):
        model = MBartForConditionalGeneration.from_pretrained(model_name_or_path)

        # net = load_ov_model_from_pytorch(model.get_encoder())
        # encoder = OVMBartEncoder(net, model.config)
        inputs = (
            None,  # input_ids
            torch.zeros([5, 18], dtype=torch.int32),  # attention_mask
            torch.zeros([5, 2], dtype=torch.int32),  # decoder_input_ids
            None,  # decoder_attention_mask
            None,  # head_mask
            None,  # decoder_head_mask
            None,  # cross_attn_head_mask
            [torch.zeros([5, 18, 1024], dtype=torch.float32)]  # encoder_outputs
        )

        net = load_ov_model_from_pytorch(model, inputs)
        model = OVPreTrainedModel(net, model.config)

        model.get_encoder = lambda: encoder

        return model

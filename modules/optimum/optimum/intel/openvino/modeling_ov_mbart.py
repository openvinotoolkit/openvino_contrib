# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from transformers.file_utils import is_torch_available

if is_torch_available():
    import torch
    from transformers import MBartForConditionalGeneration
    from transformers.modeling_outputs import BaseModelOutput

from .modeling_ov_utils import (
    OVPreTrainedModel,
    load_ov_model_from_pytorch,
    is_openvino_api_2,
)


class OVMBartEncoder(OVPreTrainedModel):
    def __init__(self, net, config):
        super().__init__(net, config)

    def __call__(self, *args, **kwargs):
        kwargs["return_dict"] = False
        res = super().__call__(*args, **kwargs)
        return BaseModelOutput(last_hidden_state=torch.tensor(res[0]))


def _prepare_nlp_inputs(
    self,
    input_ids=None,
    attention_mask=None,
    decoder_input_ids=None,
    decoder_attention_mask=None,
    head_mask=None,
    decoder_head_mask=None,
    cross_attn_head_mask=None,
    encoder_outputs=None,
    past_key_values=None,
    inputs_embeds=None,
    decoder_inputs_embeds=None,
    labels=None,
    use_cache=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
):
    return {
        "encoder_outputs": encoder_outputs.last_hidden_state,
        "attention_mask": attention_mask,
        "decoder_input_ids": decoder_input_ids,
    }


class OVMBartForConditionalGeneration(object):
    @classmethod
    def from_pretrained(cls, model_name_or_path, *model_args, **kwargs):
        kwargs.pop("from_pt", None)
        model = MBartForConditionalGeneration.from_pretrained(model_name_or_path, *model_args, **kwargs)
        if model.config.use_cache:
            raise NotImplementedError("MBart model with use_cache=True is not implemented for OpenVINO backend")

        # Origin model produces extra outputs. Return only logits.
        origin_forward = model.forward
        model.forward = lambda *args, **kwargs: origin_forward(*args, **kwargs).logits

        # Create a separate network for encoder - it will be called just once.
        net = load_ov_model_from_pytorch(model.get_encoder())
        encoder = OVMBartEncoder(net, model.config)

        inputs = {
            "input_ids": None,
            "attention_mask": torch.zeros([1, 18], dtype=torch.int32),
            "decoder_input_ids": torch.zeros([1, 18], dtype=torch.int32),
            "decoder_attention_mask": None,
            "head_mask": None,
            "decoder_head_mask": None,
            "cross_attn_head_mask": None,
            "encoder_outputs": [torch.zeros([1, 18, 1024], dtype=torch.float32)],
        }

        net = load_ov_model_from_pytorch(model, inputs)

        # Fix for 2022.1 release
        if is_openvino_api_2:
            net.inputs[2].get_tensor().set_names(set(["encoder_outputs"]))

        ov_model = OVPreTrainedModel(net, model.config)

        ov_model.get_encoder = lambda: encoder
        ov_model.prepare_inputs_for_generation = model.prepare_inputs_for_generation
        ov_model._prepare_nlp_inputs = lambda *args, **kwargs: _prepare_nlp_inputs(model, *args, **kwargs)

        return ov_model

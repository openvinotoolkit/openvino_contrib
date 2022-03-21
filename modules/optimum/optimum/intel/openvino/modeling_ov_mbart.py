# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from transformers.file_utils import is_torch_available

if is_torch_available():
    import torch
    from transformers import MBartForConditionalGeneration
    from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
    from transformers.generation_utils import GenerationMixin

from .modeling_ov_utils import (
    OVPreTrainedModel,
    load_ov_model_from_pytorch,
    is_openvino_api_2,
    ie
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

#     if "past_key_values" is not None:


    # names = ['past_key_values', '4', '3274', '3275', '7', '8', '3276', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '3288', '3289', '35', '36', '3290', '3291', '39', '40', '3292', '3293', '43', '44', '3294', '3295', '47', '48', '3296', '3297']
    # if past_key_values is not None:
    #     assert len(names) == 48
    #     for i in range(12):
    #         for j in range(4):
    #             inputs[names[0]] = past_key_values[i][j]
    #             del names[0]
    # else:
    #     for i in range(12):
    #         inputs[names[i * 4]] = np.zeros([5, 16, 0, 64], dtype=np.float32)
    #         inputs[names[i * 4 + 1]] = np.zeros([5, 16, 0, 64], dtype=np.float32)
    #         inputs[names[i * 4 + 2]] = np.zeros([5, 16, 18, 64], dtype=np.float32)
    #         inputs[names[i * 4 + 3]] = np.zeros([5, 16, 18, 64], dtype=np.float32)

    # return inputs


# A wrapper for MBart model. When use_cache=True is specified,
# first model run accepts only three inputs but next calls uses past_key_values.
# We cannot keep a single model for both cases because the graph is dynamic and connections are different.
class OVMBartForConditionalGeneration(GenerationMixin):
    def __init__(self, config, encoder, model, model_past=None):
        super().__init__()
        self.encoder = OVMBartEncoder(encoder, config)
        self.model = OVPreTrainedModel(model, config)
        # self.model_past = OVPreTrainedModel(model_past, config) if model_past else None
        # self.encoder = OVMBartEncoder(ie.read_model("encoder/ov_model.xml", "encoder/ov_model.bin"), config)
        # self.model = OVPreTrainedModel(ie.read_model("model/ov_model.xml", "model/ov_model.bin"), config)

        self.model._prepare_nlp_inputs = lambda *args, **kwargs: _prepare_nlp_inputs(self.model, *args, **kwargs)

        self.config = config
        # self.encoder.save_pretrained("encoder")
        # self.model.save_pretrained("model")
        # self.model_past.save_pretrained("model_past")
        if is_torch_available():
            self.device = torch.device("cpu")

    def get_encoder(self):
        return self.encoder

    # Copied from MBartForConditionalGeneration
    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    # Copied from MBartForConditionalGeneration
    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    @classmethod
    def from_pretrained(cls, model_name_or_path, *model_args, **kwargs):
        kwargs.pop("from_pt", None)
        model = MBartForConditionalGeneration.from_pretrained(model_name_or_path, *model_args, **kwargs)
        use_cache = model.config.use_cache
        # return OVMBartForConditionalGeneration(model.config, None, None, None)

        origin_forward = model.forward
        def forward(*args, **kwargs):
            # args = list(args)
            # args[7] *= 1  # encoder_outputs
            # args = tuple(args)
            outputs = origin_forward(*args, **kwargs)
            return Seq2SeqLMOutput(
                # loss=masked_lm_loss,
                logits=outputs.logits,
                # past_key_values=outputs.past_key_values,
            #     decoder_hidden_states=outputs.decoder_hidden_states,
            #     decoder_attentions=outputs.decoder_attentions,
            #     cross_attentions=outputs.cross_attentions,
            #     encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            #     encoder_hidden_states=outputs.encoder_hidden_states,
            #     encoder_attentions=outputs.encoder_attentions,
            )
        model.forward = lambda *args, **kwargs: forward(*args, **kwargs)

        # Create a separate network for encoder - it will be called just once.
        encoder = load_ov_model_from_pytorch(model.get_encoder())
        # encoder = OVMBartEncoder(net, model.config)

        inputs = {
            "input_ids": None,
            "attention_mask": torch.zeros([1, 11], dtype=torch.int32),
            "decoder_input_ids": torch.zeros([1, 1 if use_cache else 11], dtype=torch.int32),
            "decoder_attention_mask": None,
            "head_mask": None,
            "decoder_head_mask": None,
            "cross_attn_head_mask": None,
            "encoder_outputs": [torch.zeros([1, 11, 1024], dtype=torch.float32)],
        }

        net = load_ov_model_from_pytorch(model, inputs)

        # Fix for 2022.1 release
        if is_openvino_api_2:
            net.inputs[2].get_tensor().set_names(set(["encoder_outputs"]))

        # ov_model = OVPreTrainedModel(net, model.config)

        if use_cache:
            inputs["past_key_values"] = [[
                    torch.zeros([5, 16, 4, 64], dtype=torch.float32),
                    torch.zeros([5, 16, 4, 64], dtype=torch.float32),
                    torch.zeros([5, 16, 18, 64], dtype=torch.float32),
                    torch.zeros([5, 16, 18, 64], dtype=torch.float32),
                ]] * 12
            net_past = load_ov_model_from_pytorch(model, inputs)
        else:
            net_past = None

        return OVMBartForConditionalGeneration(model.config, encoder, net, net_past)
        # ov_model = OVMBartForConditionalGenerationUseCache(encoder, ov_model, ov_model_past)

        # ov_model.get_encoder = lambda: encoder
        # ov_model.prepare_inputs_for_generation = model.prepare_inputs_for_generation
        # ov_model._prepare_nlp_inputs = lambda *args, **kwargs: _prepare_nlp_inputs(ov_model, *args, **kwargs)
        # ov_model._reorder_cache = model._reorder_cache

        # return ov_model

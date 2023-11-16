# -*- coding: utf-8 -*-
# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import tempfile
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import openvino.runtime.opset12 as opset
from openvino import Model, PartialShape, Type
from openvino.runtime import Node, op
from openvino.runtime.exceptions import OVTypeError
from openvino.runtime.utils.types import as_node, make_constant_node
from transformers.convert_slow_tokenizer import import_protobuf

from .constants import (
    ATTENTION_MASK_INPUT_NAME,
    STRING_OUTPUT_NAME,
    TOKEN_IDS_INPUT_NAME,
    TOKEN_TYPE_IDS_INPUT_NAME,
    TOKENIZER_DECODER_NAME,
    TOKENIZER_ENCODER_NAME,
)
from .node_factory import factory
from .tokenizer_pipeline import (
    BPETokenizationStep,
    BytesToCharsStep,
    CaseFoldStep,
    CharsToBytesStep,
    CombineSegmentsStep,
    NMTNormalizationStep,
    NormalizationStep,
    NormalizeUnicode,
    PaddingStep,
    PreTokenizatinStep,
    PunctuationSplitStep,
    RegexDecodingStep,
    RegexNormalizationStep,
    RegexSplitStep,
    StripStringStep,
    TokenizerPipeline,
    TruncationStep,
    VocabDecoderStep,
    WhitespaceSplitStep,
    WordPieceTokenizationStep,
)


def parse_replace_normalizer(normalizer_dict: Dict[str, Any]) -> RegexNormalizationStep:
    regex_search_pattern = normalizer_dict["pattern"].get("String") or normalizer_dict["pattern"]["Regex"]
    return RegexNormalizationStep(
        regex_search_pattern=regex_search_pattern,
        replace_term=normalizer_dict["content"],
    )


def parse_bert_normalizer(normalizer_dict: Dict[str, Any]) -> List[NormalizationStep]:
    steps: List[NormalizationStep] = []

    if normalizer_dict["clean_text"] is True:
        steps.append(RegexNormalizationStep.del_control_chars_regex())

    # https://github.com/huggingface/tokenizers/blob/8c9cfb0b689bce00b615b9557a9a767f286d7a33/tokenizers/src/normalizers/bert.rs#L127
    if normalizer_dict.get("strip_accents") or normalizer_dict["lowercase"]:
        steps.append(NormalizeUnicode("NFD"))
        steps.append(RegexNormalizationStep.strip_accents_regex())

    if normalizer_dict["lowercase"] is True:
        steps.append(CaseFoldStep())

    return steps


def parse_strip_step(split_dict: Dict[str, Any]) -> StripStringStep:
    return StripStringStep(
        left=split_dict["strip_left"],
        right=split_dict["strip_right"],
    )


def parse_split_step(pretokenizer_dict: Dict[str, Any]) -> RegexSplitStep:
    split_pattern = pretokenizer_dict["pattern"].get("String") or pretokenizer_dict["pattern"]["Regex"]
    return RegexSplitStep(
        split_pattern=split_pattern,
        invert=pretokenizer_dict["invert"],
        behaviour=pretokenizer_dict["behavior"].lower().rstrip("d"),
    )


def parse_byte_level_pretokenization_step(
    pretokenizer_dict: Dict[str, Any]
) -> List[Union[NormalizationStep, PreTokenizatinStep]]:
    steps = []
    if pretokenizer_dict.get("add_prefix_space"):
        steps.append(RegexNormalizationStep.add_prefix_whitespace_regex())

    # regex is used by default, but it does not appear in config yet
    if pretokenizer_dict.get("use_regex", True):
        # re2 does not support negative lookahead, so there is two steps replicate the behaviour
        # this WA causes segfault for CLIP tokenizer
        # steps.append(RegexSplitStep.add_whitespace_to_the_next_word())
        steps.append(RegexSplitStep.byte_level_splitter())

    steps.append(BytesToCharsStep())
    return steps


class TransformersTokenizerPipelineParser:
    def __init__(self, tokenizer_object: Any, number_of_inputs: int = 1) -> None:
        assert tokenizer_object.is_fast

        self.original_tokenizer = tokenizer_object
        with TemporaryDirectory() as tmpdir:
            tokenizer_object.save_pretrained(tmpdir)
            with open(Path(tmpdir) / "tokenizer.json") as tj:
                self.tokenizer_json = json.load(tj)
        self.pipeline = TokenizerPipeline()
        self.number_of_inputs = number_of_inputs
        self.num_of_added_tokens = 0

    def parse(self, number_of_inputs: Optional[int] = None) -> TokenizerPipeline:
        self.number_of_inputs = self.number_of_inputs if number_of_inputs is None else number_of_inputs
        self.pipeline.number_of_inputs = self.number_of_inputs
        for add_steps in [
            self.normalization,
            self.pre_tokenization,
            self.tokenization_model,
            self.post_tokenization,
            self.decoding,
        ]:
            add_steps()

        return self.pipeline

    normalizers_map: Dict[
        str,
        Callable[[Dict[str, Any]], Union[NormalizationStep, List[NormalizationStep]]],
    ] = {
        "NFC": lambda step_dict: NormalizeUnicode("NFC"),
        "NFD": lambda step_dict: NormalizeUnicode("NFD"),
        "NFKC": lambda step_dict: NormalizeUnicode("NFKC"),
        "NFKD": lambda step_dict: NormalizeUnicode("NFKD"),
        "Nmt": lambda step_dict: NMTNormalizationStep(),
        "Lowercase": lambda step_dict: CaseFoldStep(),
        "StripAccents": lambda step_dict: RegexNormalizationStep.strip_accents_regex(),
        "BertNormalizer": parse_bert_normalizer,
        "Replace": parse_replace_normalizer,
        "Strip": parse_strip_step,
    }

    def parse_normalizer_step(self, step_dict: Dict[str, Any]) -> None:
        try:
            self.pipeline.add_steps(self.normalizers_map[step_dict["type"]](step_dict))
        except KeyError:
            raise OVTypeError(f"Normalizer type '{step_dict['type']}' is not supported")

    def normalization(self) -> None:
        if self.tokenizer_json["normalizer"] is None:
            return

        if self.tokenizer_json["normalizer"].get("type") == "Sequence":
            for normalizer in self.tokenizer_json["normalizer"]["normalizers"]:
                self.parse_normalizer_step(normalizer)
        else:
            self.parse_normalizer_step(self.tokenizer_json["normalizer"])

    pre_tokenization_map: Dict[
        str,
        Callable[[Dict[str, Any]], Union[PreTokenizatinStep, List[PreTokenizatinStep]]],
    ] = {
        "BertPreTokenizer": lambda step_dict: RegexSplitStep.bert_splitter(),
        "Whitespace": lambda step_dict: RegexSplitStep.whitespace_splitter(),
        "WhitespaceSplit": lambda step_dict: WhitespaceSplitStep(),
        "Split": parse_split_step,
        "Punctuation": lambda step_dict: PunctuationSplitStep(step_dict["behavior"]),
        "ByteLevel": parse_byte_level_pretokenization_step,
        "Digits": lambda step_dict: RegexSplitStep.digits_splitter(
            "isolate" if step_dict["individual_digits"] else "contiguous"
        ),
    }

    def parse_pre_tokenization_step(self, step_dict: Dict[str, Any]) -> None:
        try:
            self.pipeline.add_steps(self.pre_tokenization_map[step_dict["type"]](step_dict))
        except KeyError:
            raise OVTypeError(f"Pre-tokenizer type '{step_dict['type']}' is not supported")

    def pre_tokenization(self) -> None:
        if self.tokenizer_json["pre_tokenizer"] is None:
            return

        if self.tokenizer_json["pre_tokenizer"].get("type") == "Sequence":
            for pretokenizer in self.tokenizer_json["pre_tokenizer"]["pretokenizers"]:
                self.parse_pre_tokenization_step(pretokenizer)
        else:
            self.parse_pre_tokenization_step(self.tokenizer_json["pre_tokenizer"])

    def tokenization_model(self) -> None:
        if self.tokenizer_json["model"]["type"] == "WordPiece":
            self.pipeline.add_steps(WordPieceTokenizationStep.from_hf_json(self.tokenizer_json))
            self.pipeline.vocab = self.pipeline[-1].vocab
        elif self.tokenizer_json["model"]["type"] == "BPE":
            self.pipeline.add_steps(BPETokenizationStep.from_hf_json(self.tokenizer_json))
            self.pipeline.vocab = self.pipeline[-1].vocab
        else:
            raise OVTypeError(f"Tokenizer type '{self.tokenizer_json['model']['type']}' is not supported")

    def post_tokenization(self) -> None:
        if (
            self.tokenizer_json["post_processor"] is None
            or self.tokenizer_json["post_processor"]["type"] == "ByteLevel"
        ):
            self.add_truncation()
            self.add_padding()
            return

        if self.tokenizer_json["post_processor"]["type"] == "TemplateProcessing":
            combine_segments_step = CombineSegmentsStep.from_hf_json_template_postprocessor(
                self.tokenizer_json, self.number_of_inputs
            )
        elif self.tokenizer_json["post_processor"]["type"] == "BertProcessing":
            combine_segments_step = CombineSegmentsStep.from_hf_json_bert_postprocessor(
                self.tokenizer_json, self.number_of_inputs
            )
        elif self.tokenizer_json["post_processor"]["type"] == "RobertaProcessing":
            combine_segments_step = CombineSegmentsStep.from_hf_json_roberta_processor(
                self.tokenizer_json, self.number_of_inputs
            )
        else:
            raise OVTypeError(
                f"Post-processor type '{self.tokenizer_json['post_processor']['type']}' is not supported"
            )

        self.num_of_added_tokens += combine_segments_step.number_of_added_tokens
        combine_segments_step.set_tokens_ids(self.pipeline.vocab)

        self.add_truncation()
        self.pipeline.add_steps(combine_segments_step)

        self.add_padding()

    def add_truncation(self) -> None:
        if self.tokenizer_json["truncation"] is not None:
            self.pipeline.add_steps(TruncationStep.from_hf_json(self.tokenizer_json, self.num_of_added_tokens))
        elif self.original_tokenizer.model_max_length is not None:
            self.pipeline.add_steps(TruncationStep.from_hf_object(self.original_tokenizer, self.num_of_added_tokens))

    def add_padding(self) -> None:
        if self.tokenizer_json["padding"] is not None:
            self.pipeline.add_steps(PaddingStep.from_hf_json(self.tokenizer_json))
            self.pipeline[-1].set_token_id(self.pipeline.vocab)
        elif self.original_tokenizer.pad_token is not None:
            self.pipeline.add_steps(PaddingStep(token=self.original_tokenizer.pad_token))
            self.pipeline[-1].set_token_id(self.pipeline.vocab)
        else:
            self.pipeline.add_steps(PaddingStep())

    def decoding(self) -> None:
        if self.tokenizer_json["decoder"] is None:
            return

        if self.tokenizer_json["decoder"]["type"] == "ByteLevel":
            self.pipeline.add_steps(VocabDecoderStep())
            self.pipeline.add_steps(CharsToBytesStep())

        if self.original_tokenizer.clean_up_tokenization_spaces:
            self.pipeline.add_steps(RegexDecodingStep.clean_up_tokenization_spaces())
        return


def convert_fast_tokenizer(
    hf_tokenizer: "PreTrainedTokenizerBase",
    number_of_inputs: int = 1,
    with_decoder: bool = False,
) -> Union[Model, Tuple[Model, Model]]:
    pipeline = TransformersTokenizerPipelineParser(hf_tokenizer).parse(number_of_inputs=number_of_inputs)
    ov_tokenizer = pipeline.get_encoder_ov_subgraph()
    output_names = hf_tokenizer.model_input_names

    ov_tokenizer_output_names = [TOKEN_IDS_INPUT_NAME, ATTENTION_MASK_INPUT_NAME]
    if len(output_names) == 3 and len(ov_tokenizer.outputs) == 3:
        ov_tokenizer_output_names.insert(1, TOKEN_TYPE_IDS_INPUT_NAME)

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

    tokenizer_model = Model(filtered_outputs, ov_tokenizer.get_parameters(), TOKENIZER_ENCODER_NAME)
    if with_decoder:
        return tokenizer_model, pipeline.get_decoder_ov_subgraph()

    return tokenizer_model


def is_sentencepiece_model(hf_tokenizer: "PreTrainedTokenizerBase") -> bool:
    return getattr(hf_tokenizer, "vocab_files_names", {}).get("vocab_file", "").endswith(".model")


def add_tokens_to_sentencepiece_model(sp_model_path: Path, hf_tokenizer: "PreTrainedTokenizerBase") -> None:
    model_pb = import_protobuf()
    model = model_pb.ModelProto()
    with open(sp_model_path, "rb") as model_file:
        model.ParseFromString(model_file.read())

    add_token_dict = hf_tokenizer.tokenizer.index_special_tokens
    for idx, token in sorted(add_token_dict.items()):
        new_piece = deepcopy(model.pieces[-1])
        new_piece.piece = token
        model.pieces.append(new_piece)

    with open(sp_model_path, "wb") as model_file:
        model_file.write(model.SerializeToString())


def convert_sentencepiece_model_tokenizer(
    hf_tokenizer: "PreTrainedTokenizerBase",
    add_attention_mask: bool = True,
    with_decoder: bool = False,
    streaming_decoder: bool = False,
) -> Union[Model, Tuple[Model, Model]]:
    if not is_sentencepiece_model(hf_tokenizer):
        raise OVTypeError("Cannot convert tokenizer that does not have `.model` file.")

    fairseq_offset = getattr(hf_tokenizer, "fairseq_offset", None)

    with tempfile.TemporaryDirectory() as tmp:
        hf_tokenizer.save_pretrained(tmp)
        vocab_file = Path(tmp) / hf_tokenizer.vocab_files_names["vocab_file"]

        if (is_chatglm := getattr(hf_tokenizer, "name", None) == "GLMTokenizer"):
            add_tokens_to_sentencepiece_model(vocab_file, hf_tokenizer)

        sp_model = np.fromfile(vocab_file, dtype=np.uint8)
        sp_model_node = as_node(sp_model)

        if hf_tokenizer.is_fast:
            hf_slow_tokenizer = hf_tokenizer.slow_tokenizer_class.from_pretrained(tmp)
            fairseq_offset = getattr(hf_slow_tokenizer, "fairseq_offset", None)

    input_node = op.Parameter(Type.u8, PartialShape(["?"]))
    input_node.set_friendly_name("string_input")

    if is_chatglm:
        add_eos_token = False
    elif hasattr(hf_tokenizer, "add_eos_token"):
        add_eos_token = hf_tokenizer.add_eos_token or False
    else:
        add_eos_token = (
            getattr(hf_tokenizer, "truncation_side", "") == "right"
            or getattr(hf_tokenizer, "padding_side", "") == "right"
        )
    add_bos_token = getattr(hf_tokenizer, "add_bos_token", add_eos_token) or False

    tokenizer_node = factory.create(
        "SentencepieceTokenizer",
        [sp_model_node, input_node],
        {
            "add_bos": add_bos_token,
            "add_eos": add_eos_token,
            "reverse": False,
            "alpha": 0.0,
        },
    )

    indices, values, dense_shape = tokenizer_node.outputs()

    if fairseq_offset:
        values = opset.add(values, make_constant_node(fairseq_offset, values.element_type)).output(0)

    default_value = make_constant_node(hf_tokenizer.pad_token_id or 0, values.element_type)
    broadcast = opset.broadcast(default_value, dense_shape)
    scatternd_input_ids = factory.create(
        "ScatterNDUpdate",
        [broadcast, indices, values],  # FIXME: pad left side instead of right
    )

    if is_chatglm:
        prefix_tokens = make_constant_node(
            np.array([hf_tokenizer.get_prefix_tokens()]), dtype=scatternd_input_ids.output(0).element_type
        )
        scatternd_input_ids = opset.concat([prefix_tokens, scatternd_input_ids], axis=-1)

    scatternd_input_ids.output(0).tensor.add_names({TOKEN_IDS_INPUT_NAME})

    outputs = scatternd_input_ids.outputs()

    if add_attention_mask:
        attention_mask = factory.create(
            "ScatterNDUpdate",
            [
                broadcast,
                indices,
                opset.broadcast(
                    make_constant_node(1, values.element_type),
                    opset.shape_of(values),
                ),
            ],
        )

        if is_chatglm:
            attention_prefix = make_constant_node(
                np.array([[1 for _ in hf_tokenizer.get_prefix_tokens()]]), dtype=attention_mask.output(0).element_type
            )
            attention_mask = opset.concat([attention_prefix, attention_mask], axis=-1)

        attention_mask.output(0).tensor.add_names({ATTENTION_MASK_INPUT_NAME})
        outputs.append(attention_mask.output(0))

    tokenizer_encoder = Model(outputs, [input_node], TOKENIZER_ENCODER_NAME)
    tokenizer_encoder.validate_nodes_and_infer_types()

    if not with_decoder:
        return tokenizer_encoder

    return tokenizer_encoder, get_sp_decoder(sp_model_node, streaming_decoder=streaming_decoder)


def get_sp_decoder(sp_model_node: Node, streaming_decoder: bool = False) -> Model:
    token_ids = op.Parameter(Type.i32, PartialShape(["?", "?"]))  # (batch, sequence)

    decoder = factory.create(
        "SentencepieceStreamDetokenizer" if streaming_decoder else "SentencepieceDetokenizer",
        [sp_model_node, token_ids],
    ).outputs()

    string_output = factory.create("StringTensorPack", decoder).outputs()
    string_output[0].tensor.add_names({STRING_OUTPUT_NAME})
    tokenizer_decoder = Model(string_output, [token_ids], TOKENIZER_DECODER_NAME)
    tokenizer_decoder.validate_nodes_and_infer_types()
    return tokenizer_decoder


def is_tiktoken_model(hf_tokenizer: "PreTrainedTokenizerBase") -> bool:
    try:
        from tiktoken import Encoding
    except ImportError:
        return False

    return getattr(hf_tokenizer, "vocab_files_names", {}).get("vocab_file", "").endswith(".tiktoken") or isinstance(
        getattr(hf_tokenizer, "encoder", None), Encoding
    )


def convert_tiktoken_model_tokenizer(
    hf_tokenizer: "PreTrainedTokenizerBase",
    with_decoder: bool = False,
) -> Union[Model, Tuple[Model, Model]]:
    encoding = getattr(hf_tokenizer, "tokenizer", None) or hf_tokenizer.encoder
    split_pattern = encoding._pat_str

    pipeline = TokenizerPipeline()
    pipeline.add_steps(
        [
            NormalizeUnicode("NFC"),
            RegexSplitStep(split_pattern),
            BytesToCharsStep(),
            BPETokenizationStep.from_tiktoken_encoding(encoding),
            TruncationStep(
                max_length=hf_tokenizer.model_max_length, truncate_right=(hf_tokenizer.truncation_side == "right")
            ),
            PaddingStep(pad_right=(hf_tokenizer.padding_side == "right")),
            VocabDecoderStep(),
            CharsToBytesStep(),
        ]
    )
    if not with_decoder:
        return pipeline.get_encoder_ov_subgraph()

    return pipeline.get_encoder_ov_subgraph(), pipeline.get_decoder_ov_subgraph()

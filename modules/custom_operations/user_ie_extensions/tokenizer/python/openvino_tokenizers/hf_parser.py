# -*- coding: utf-8 -*-
# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import tempfile
from copy import deepcopy
from functools import partial
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

from . import _get_factory
from .constants import (
    ATTENTION_MASK_INPUT_NAME,
    DETOKENIZER_NAME,
    STRING_OUTPUT_NAME,
    TOKEN_IDS_INPUT_NAME,
    TOKEN_TYPE_IDS_INPUT_NAME,
    TOKENIZER_NAME,
)
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
        if not tokenizer_object.is_fast:
            raise OVTypeError("Tokenizer is not supported.")

        self.original_tokenizer = tokenizer_object
        with TemporaryDirectory() as tmpdir:
            tokenizer_object.save_pretrained(tmpdir)
            # Windows uses cp1252 encoding by default, need to use utf-8 explicitly
            with open(Path(tmpdir) / "tokenizer.json", encoding="utf-8") as tj:
                self.tokenizer_json = json.load(tj)
        self.pipeline = TokenizerPipeline()
        self.number_of_inputs = number_of_inputs
        self.num_of_added_tokens = 0

    def parse(
        self,
        number_of_inputs: Optional[int] = None,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: Optional[bool] = None,
    ) -> TokenizerPipeline:
        self.number_of_inputs = self.number_of_inputs if number_of_inputs is None else number_of_inputs
        self.pipeline.number_of_inputs = self.number_of_inputs
        for add_steps in [
            self.normalization,
            self.pre_tokenization,
            self.tokenization_model,
            self.post_tokenization,
            partial(
                self.decoding,
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            ),
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

    def decoding(
        self,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: Optional[bool] = None,
    ) -> None:
        if self.tokenizer_json["decoder"] is None:
            return

        skip_tokens = parse_special_tokens(self.original_tokenizer) if skip_special_tokens else {}
        if self.tokenizer_json["decoder"]["type"] == "ByteLevel":
            self.pipeline.add_steps(VocabDecoderStep(list(skip_tokens)))
            self.pipeline.add_steps(CharsToBytesStep())

        if suffix := self.tokenizer_json["model"].get("end_of_word_suffix"):
            self.pipeline.add_steps(RegexDecodingStep.replace_end_of_word_suffix(suffix))

        if prefix := self.tokenizer_json["model"].get("continuing_subword_prefix"):
            self.pipeline.add_steps(RegexDecodingStep.replace_continuing_subword_prefix(prefix))

        if clean_up_tokenization_spaces is None:
            clean_up_tokenization_spaces = self.original_tokenizer.clean_up_tokenization_spaces

        if clean_up_tokenization_spaces and self.pipeline.decoding_steps:
            self.pipeline.add_steps(RegexDecodingStep.clean_up_tokenization_spaces())
        return


def parse_special_tokens(hf_tokenizer: "PreTrainedTokenizerBase") -> Dict[int, str]:
    # the order matters
    if getattr(hf_tokenizer, "added_tokens_decoder", False):
        return {
            idx: added_token.content
            for idx, added_token in hf_tokenizer.added_tokens_decoder.items()
            if added_token.special
        }
    elif getattr(hf_tokenizer, "tokenizer", False) and getattr(hf_tokenizer.tokenizer, "index_special_tokens", False):
        return hf_tokenizer.tokenizer.index_special_tokens
    elif getattr(hf_tokenizer, "special_tokens", False):
        return {idx: token for token, idx in sorted(hf_tokenizer.special_tokens.items(), key=lambda x: x[1])}

    return {}


def convert_fast_tokenizer(
    hf_tokenizer: "PreTrainedTokenizerBase",
    number_of_inputs: int = 1,
    with_detokenizer: bool = False,
    skip_special_tokens: bool = False,
    clean_up_tokenization_spaces: Optional[bool] = None,
) -> Union[Model, Tuple[Model, Model]]:
    pipeline = TransformersTokenizerPipelineParser(hf_tokenizer).parse(
        number_of_inputs=number_of_inputs,
        skip_special_tokens=skip_special_tokens,
        clean_up_tokenization_spaces=clean_up_tokenization_spaces,
    )
    ov_tokenizer = pipeline.get_tokenizer_ov_subgraph()
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

    tokenizer_model = Model(filtered_outputs, ov_tokenizer.get_parameters(), TOKENIZER_NAME)
    if with_detokenizer:
        return tokenizer_model, pipeline.get_detokenizer_ov_subgraph()

    return tokenizer_model


def is_sentencepiece_model(hf_tokenizer: "PreTrainedTokenizerBase") -> bool:
    return getattr(hf_tokenizer, "vocab_files_names", {}).get("vocab_file", "").endswith(".model")


def modify_sentencepiece_model(
    sp_model_path: Path,
    add_tokens: Dict[int, str],
    skip_special_tokens: bool = False,
    reference_vocab: Optional[List[str]] = None,
) -> None:
    model_pb = import_protobuf()
    model = model_pb.ModelProto()
    with open(sp_model_path, "rb") as model_file:
        model.ParseFromString(model_file.read())

    existing = {piece.piece: piece for piece in model.pieces}
    for idx, token in sorted(add_tokens.items()):
        if to_add := (idx >= len(model.pieces) or model.pieces[idx].piece != token):
            if exists := existing.get(token):
                new_piece = model.pieces.pop(next(idx for idx, piece in enumerate(model.pieces) if piece == exists))
            else:
                new_piece = deepcopy(model.pieces[-1])
                new_piece.piece = token
        else:
            new_piece = model.pieces[idx]

        if skip_special_tokens and new_piece.type != 2:  # type 2 is for unk symbol
            new_piece.type = 3  # make it control symbol so it will not decode during detokenization
        elif not skip_special_tokens and new_piece.type == 3:
            new_piece.type = 4  # change control type to userdef type

        if to_add:
            model.pieces.insert(idx, new_piece)

    # change unk token representation from ⁇ to token string
    unk_token = next(piece for piece in model.pieces if piece.type == 2)
    model.trainer_spec.unk_surface = unk_token.piece

    with open(sp_model_path, "wb") as model_file:
        model_file.write(model.SerializeToString())


def convert_sentencepiece_model_tokenizer(
    hf_tokenizer: "PreTrainedTokenizerBase",
    add_attention_mask: bool = True,
    with_detokenizer: bool = False,
    streaming_detokenizer: bool = False,
    skip_special_tokens: bool = False,
    clean_up_tokenization_spaces: Optional[bool] = False,
) -> Union[Model, Tuple[Model, Model]]:
    if not is_sentencepiece_model(hf_tokenizer):
        raise OVTypeError("Cannot convert tokenizer that does not have `.model` file.")

    with tempfile.TemporaryDirectory() as tmp:
        hf_tokenizer.save_pretrained(tmp)
        vocab_file = Path(tmp) / hf_tokenizer.vocab_files_names["vocab_file"]

        add_tokens = parse_special_tokens(hf_tokenizer)
        modify_sentencepiece_model(
            sp_model_path=vocab_file,
            add_tokens=add_tokens,
            skip_special_tokens=skip_special_tokens,
        )

        sp_model = np.fromfile(vocab_file, dtype=np.uint8)
        sp_model_node = as_node(sp_model)

    input_node = op.Parameter(Type.string, PartialShape(["?"]))
    input_node.set_friendly_name("string_input")

    is_chatglm = getattr(hf_tokenizer, "name", None) == "GLMTokenizer"
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

    tokenizer_node = _get_factory().create(
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

    default_value = make_constant_node(hf_tokenizer.pad_token_id or 0, values.element_type)
    broadcast = opset.broadcast(default_value, dense_shape)
    scatternd_input_ids = _get_factory().create(
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
        attention_mask = _get_factory().create(
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

    tokenizer = Model(outputs, [input_node], TOKENIZER_NAME)
    tokenizer.validate_nodes_and_infer_types()

    if not with_detokenizer:
        return tokenizer

    if clean_up_tokenization_spaces is None:
        clean_up_tokenization_spaces = hf_tokenizer.clean_up_tokenization_spaces

    return tokenizer, get_sp_detokenizer(
        sp_model_node,
        streaming_detokenizer=streaming_detokenizer,
        clean_up_tokenization_spaces=clean_up_tokenization_spaces,
    )


def get_sp_detokenizer(
    sp_model_node: Node, streaming_detokenizer: bool = False, clean_up_tokenization_spaces: bool = False
) -> Model:
    model_input = token_ids = op.Parameter(Type.i32, PartialShape(["?", "?"]))  # (batch, sequence)

    detokenizer = (
        _get_factory()
        .create(
            "SentencepieceStreamDetokenizer" if streaming_detokenizer else "SentencepieceDetokenizer",
            [sp_model_node, token_ids],
        )
        .outputs()
    )

    if streaming_detokenizer:
        detokenizer = RegexDecodingStep.replace_sp_spaces().get_ov_subgraph(detokenizer)

    if clean_up_tokenization_spaces:
        detokenizer = RegexDecodingStep.clean_up_tokenization_spaces().get_ov_subgraph(detokenizer)

    string_output = _get_factory().create("StringTensorPack", detokenizer).outputs()
    string_output[0].tensor.add_names({STRING_OUTPUT_NAME})
    tokenizer_detokenizer = Model(string_output, [model_input], DETOKENIZER_NAME)
    tokenizer_detokenizer.validate_nodes_and_infer_types()
    return tokenizer_detokenizer


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
    with_detokenizer: bool = False,
    skip_special_tokens: bool = False,
    clean_up_tokenization_spaces: Optional[bool] = None,
) -> Union[Model, Tuple[Model, Model]]:
    encoding = getattr(hf_tokenizer, "tokenizer", None) or hf_tokenizer.encoder
    split_pattern = encoding._pat_str

    pipeline = TokenizerPipeline()
    skip_tokens = []
    if skip_special_tokens:
        skip_tokens = list(parse_special_tokens(hf_tokenizer))

    pipeline.add_steps(
        [
            NormalizeUnicode("NFC"),
            RegexSplitStep(split_pattern),
            BytesToCharsStep(),
            BPETokenizationStep.from_tiktoken_encoding(encoding),
            TruncationStep.from_hf_object(hf_tokenizer),
            PaddingStep(pad_right=(hf_tokenizer.padding_side == "right")),
            VocabDecoderStep(skip_tokens),
            CharsToBytesStep(),
        ]
    )
    if clean_up_tokenization_spaces is None:
        clean_up_tokenization_spaces = getattr(hf_tokenizer, "clean_up_tokenization_spaces", None)

    if clean_up_tokenization_spaces:
        pipeline.add_steps(RegexDecodingStep.clean_up_tokenization_spaces())

    if not with_detokenizer:
        return pipeline.get_tokenizer_ov_subgraph()

    return pipeline.get_tokenizer_ov_subgraph(), pipeline.get_detokenizer_ov_subgraph()

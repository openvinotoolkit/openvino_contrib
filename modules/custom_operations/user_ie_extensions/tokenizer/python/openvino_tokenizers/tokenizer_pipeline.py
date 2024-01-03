# -*- coding: utf-8 -*-
# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import weakref
from dataclasses import dataclass, field
from functools import singledispatchmethod
from itertools import chain, islice
from typing import Any, Dict, List, Optional, Union

import numpy as np
from openvino.runtime import Model, Output, PartialShape, Type, op
from openvino.runtime import opset12 as opset
from openvino.runtime.exceptions import OVTypeError, UserInputError
from openvino.runtime.utils.types import as_node, make_constant_node

from . import _get_factory
from .constants import (
    ATTENTION_MASK_INPUT_NAME,
    DETOKENIZER_NAME,
    STRING_OUTPUT_NAME,
    TOKEN_IDS_INPUT_NAME,
    TOKEN_TYPE_IDS_INPUT_NAME,
    TOKENIZER_NAME,
)
from .str_pack import pack_string, pack_strings


class BasePipelineStep:
    _pipeline = field(default=None, init=False, repr=False)

    def __str__(self) -> str:
        params_string = ", ".join(f"{key}={val!r}" for key, val in self.get_config().items())
        return f"{self.__class__.__name__}({params_string})"

    def get_config(self) -> Dict[str, Any]:
        config = {key: value for key, value in vars(self).items() if not key.startswith("_")}
        properties = {
            key: getattr(self, key)
            for key in dir(type(self))
            if not key.startswith("_") and isinstance(getattr(type(self), key), property)
        }
        config.update(properties)
        return config

    def get_pipeline(self) -> Optional["TokenizerPipeline"]:
        return self._pipeline()

    def set_pipeline(self, pipeline: "TokenizerPipeline") -> None:
        self._pipeline = weakref.ref(pipeline)

    def get_ov_subgraph(self, *input_nodes: List[Output]) -> List[Output]:
        raise NotImplementedError

    @staticmethod
    def create_string_constant_node(value: Union[str, List[str]]) -> op.Constant:
        if isinstance(value, str):
            # string scalar
            ps = pack_string(value)
            return op.Constant(ps)
        else:
            # support only 1D strings for now
            ps = pack_strings(value)
            return _get_factory().create("StringTensorUnpack", op.Constant(ps).outputs())


@dataclass
class NormalizationStep(BasePipelineStep):
    pass


@dataclass
class NormalizeUnicode(NormalizationStep):
    normalization_form: str = "NFD"

    def get_ov_subgraph(self, input_nodes: List[Output]) -> List[Output]:
        return (
            _get_factory()
            .create(
                "NormalizeUnicode",
                input_nodes,
                {"normalization_form": self.normalization_form},
            )
            .outputs()
        )


@dataclass
class CaseFoldStep(NormalizationStep):
    def get_ov_subgraph(self, input_nodes: List[Output]) -> List[Output]:
        return _get_factory().create("CaseFold", input_nodes).outputs()


@dataclass
class RegexNormalizationStep(NormalizationStep):
    regex_search_pattern: str
    replace_term: str

    @classmethod
    def strip_accents_regex(cls) -> "RegexNormalizationStep":
        return cls(regex_search_pattern=r"\p{Mn}", replace_term="")

    @classmethod
    def add_prefix_whitespace_regex(cls) -> "RegexNormalizationStep":
        return cls(regex_search_pattern=r"^(\S)", replace_term=r" \1")

    @classmethod
    def del_control_chars_regex(cls) -> "RegexNormalizationStep":
        # https://github.com/huggingface/tokenizers/blob/8c9cfb0b689bce00b615b9557a9a767f286d7a33/tokenizers/src/normalizers/bert.rs#L17
        return cls(
            regex_search_pattern=r"((?=[^\n\t\r])\p{Cc})|((?=[^\n\t\r])\p{Cf})",
            replace_term=" ",
        )

    @classmethod
    def clean_up_tokenization_spaces(cls) -> "RegexNormalizationStep":
        return cls(
            regex_search_pattern=r" ([\.\?\!\,])| ('[ms])| (') | ('[rv]e)",
            replace_term="\1",
        )

    def get_ov_subgraph(self, input_nodes: List[Output]) -> List[Output]:
        input_nodes.extend(
            (
                self.create_string_constant_node(self.regex_search_pattern),
                self.create_string_constant_node(self.replace_term),
            )
        )
        return _get_factory().create("RegexNormalization", input_nodes).outputs()


@dataclass
class NMTNormalizationStep(NormalizationStep):
    """Normaization based on NMT task.

    https://github.com/huggingface/tokenizers/blob/28cd3dce2a75d106572392194ff2564574c33235/tokenizers/src/normalizers/unicode.rs#L44
    """


@dataclass
class StripAccentsStep(NormalizationStep):
    def get_ov_subgraph(self, input_nodes: List[Output]) -> List[Output]:
        return RegexNormalizationStep.strip_accents_regex().get_ov_subgraph(input_nodes).outputs()


@dataclass
class DelControlCharsStep(NormalizationStep):
    def get_ov_subgraph(self, input_nodes: List[Output]) -> List[Output]:
        return RegexNormalizationStep.del_control_chars_regex().get_ov_subgraph(input_nodes).outputs()


@dataclass
class StripStringStep(NormalizationStep):
    left: bool
    right: bool


@dataclass
class PreTokenizatinStep(BasePipelineStep):
    pass


@dataclass
class RegexSplitStep(PreTokenizatinStep):
    split_pattern: str
    invert: bool = False
    behaviour: str = "remove"

    @classmethod
    def bert_whitespace_splitter(cls) -> "RegexSplitStep":
        return cls(split_pattern=r"\s+", invert=False)

    @classmethod
    def bert_keep_delimeters_splitter(cls) -> "RegexSplitStep":
        """Generates a step with a standard BERT regex.

        The source:
        https://github.com/tensorflow/text/blob/4a098cd852c0b7ebee621e2d211c7f202dd679c2/tensorflow_text/python/ops/bert_tokenizer.py#L39
        """
        return cls(
            "|".join(
                [
                    r"|".join(
                        [
                            r"[!-/]",
                            r"[:-@]",
                            r"[\[-`]",
                            r"[{-~]",
                            r"[\p{P}]",
                        ],
                    ),
                    r"|".join(
                        [
                            r"[\x{4E00}-\x{9FFF}]",
                            r"[\x{3400}-\x{4DBF}]",
                            r"[\x{20000}-\x{2A6DF}]",
                            r"[\x{2A700}-\x{2B73F}]",
                            r"[\x{2B740}-\x{2B81F}]",
                            r"[\x{2B820}-\x{2CEAF}]",
                            r"[\x{F900}-\x{FAFF}]",
                            r"[\x{2F800}-\x{2FA1F}]",
                        ],
                    ),
                ],
            ),
            invert=False,
            behaviour="isolate",
        )

    @classmethod
    def bert_splitter(cls) -> List["RegexSplitStep"]:
        return [cls.bert_whitespace_splitter(), cls.bert_keep_delimeters_splitter()]

    @classmethod
    def whitespace_splitter(cls) -> "RegexSplitStep":
        return cls(r"\w+|[^\w\s]+", invert=True)

    @classmethod
    def byte_level_splitter(cls) -> "RegexSplitStep":
        return cls(
            r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+",
            invert=False,
            behaviour="isolate",
        )

    @classmethod
    def add_whitespace_to_the_next_word(cls):
        return cls(r"\s\S", invert=False, behaviour="merge_with_next")

    @classmethod
    def digits_splitter(cls, behaviour="isolate") -> "RegexSplitStep":
        return cls(
            r"\p{Nd}|\p{Nl}|\p{No}",
            invert=False,
            behaviour=behaviour,
        )

    def get_ov_subgraph(self, input_nodes: List[Output]) -> List[Output]:
        input_nodes.extend(self.create_string_constant_node(self.split_pattern).outputs())
        return (
            _get_factory()
            .create(
                "RegexSplit",
                input_nodes,
                {
                    "behaviour": self.behaviour.lower(),
                    "invert": self.invert,
                },
            )
            .outputs()
        )


@dataclass
class WhitespaceSplitStep(PreTokenizatinStep):
    """Works like python `str.split`."""

    def get_ov_subgraph(self, input_nodes: List[Output]) -> List[Output]:
        return RegexSplitStep.whitespace_splitter().get_ov_subgraph(input_nodes).outputs()


@dataclass
class PunctuationSplitStep(PreTokenizatinStep):
    """Splits string on punctuation chars."""

    # behaviour: str = "Isolated"


@dataclass
class BytesToCharsStep(PreTokenizatinStep):
    """Maps chars to other chars for Byte-level BPE Tokenizer"""

    def get_ov_subgraph(self, input_nodes: List[Output]) -> List[Output]:
        return (
            _get_factory()
            .create(
                "BytesToChars",
                input_nodes,
            )
            .outputs()
        )


@dataclass
class TokenizationModelStep(BasePipelineStep):
    pass


@dataclass
class WordPieceTokenizationStep(TokenizationModelStep):
    vocab: List[str] = field(repr=False)
    unk_token: str = "[UNK]"
    suffix_indicator: str = "##"
    max_bytes_per_word: int = 100
    unk_token_id: int = field(init=False)

    def __post_init__(self) -> None:
        try:
            self.unk_token_id = self.vocab.index(self.unk_token)
        except ValueError:
            raise UserInputError(f"Cannot find unknown token '{self.unk_token}' in the vocab")

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @classmethod
    def from_hf_json(cls, tokenizer_json: Dict[str, Any]) -> "WordPieceTokenizationStep":
        return cls(
            unk_token=tokenizer_json["model"]["unk_token"],
            suffix_indicator=tokenizer_json["model"]["continuing_subword_prefix"],
            vocab=[token for token, index in sorted(tokenizer_json["model"]["vocab"].items(), key=lambda x: x[1])],
        )

    def get_ov_subgraph(self, input_nodes: List[Output]) -> List[Output]:
        input_nodes.extend(
            (
                *self.create_string_constant_node(self.vocab).outputs(),
                *as_node(self.unk_token_id).outputs(),
            )
        )
        return (
            _get_factory()
            .create(
                "WordpieceTokenizer",
                input_nodes,
                {
                    "suffix_indicator": self.suffix_indicator,
                    "max_bytes_per_word": self.max_bytes_per_word,
                },
            )
            .outputs()
        )


@dataclass
class BPETokenizationStep(TokenizationModelStep):
    vocab: List[str] = field(repr=False)
    merges: List[str] = field(repr=False)
    unk_token: str = ""
    fuse_unk: bool = False
    suffix_indicator: str = ""
    end_suffix: str = ""
    byte_fallback: bool = False
    added_tokens: Optional[Dict[int, str]] = None

    def __post_init__(self):
        if self.added_tokens is not None:
            self.extend_vocab_with_added_tokens()

    def extend_vocab_with_added_tokens(self) -> None:
        for idx, token in sorted(self.added_tokens.items()):
            self.vocab.append(token)

    @classmethod
    def from_hf_json(cls, tokenizer_json: Dict[str, Any]) -> "BPETokenizationStep":
        vocab = [token for token, index in sorted(tokenizer_json["model"]["vocab"].items(), key=lambda x: x[1])]
        return cls(
            unk_token=tokenizer_json["model"]["unk_token"] or "",
            fuse_unk=tokenizer_json["model"]["fuse_unk"] or False,
            suffix_indicator=tokenizer_json["model"]["continuing_subword_prefix"] or "",
            end_suffix=tokenizer_json["model"]["end_of_word_suffix"] or "",
            vocab=vocab,
            merges=tokenizer_json["model"]["merges"],
            added_tokens={
                token["id"]: token["content"] for token in tokenizer_json["added_tokens"] if token["id"] >= len(vocab)
            },
        )

    @classmethod
    def from_tiktoken_encoding(
        cls,
        encoding: "Encoding",  # noqa
        added_tokens: Optional[Dict[int, str]] = None,
    ) -> "BPETokenizationStep":
        from .tiktoken_parser import generate_vocab_and_merges

        vocab, merges = generate_vocab_and_merges(encoding)
        return cls(
            unk_token="",
            fuse_unk=False,
            suffix_indicator="",
            end_suffix="",
            vocab=[token for token, idx in sorted(vocab.items(), key=lambda x: x[1])],
            merges=merges,
            added_tokens=added_tokens,
        )

    def get_ov_subgraph(self, input_nodes: List[Output]) -> List[Output]:
        pipeline = self.get_pipeline()
        pipeline.vocab_node_outputs = self.create_string_constant_node(self.vocab).outputs()
        input_nodes.extend(
            (
                *self.get_pipeline().vocab_node_outputs,
                *self.create_string_constant_node(self.merges).outputs(),
            )
        )
        return (
            _get_factory()
            .create(
                "BPETokenizer",
                input_nodes,
                {
                    "unk_token": self.unk_token,
                    "fuse_unk": self.fuse_unk,
                    "suffix_indicator": self.suffix_indicator,
                    "end_suffix": self.end_suffix,
                    "byte_fallback": self.byte_fallback,
                },
            )
            .outputs()
        )


@dataclass
class PostTokenizationStep(BasePipelineStep):
    pass


@dataclass
class TruncationStep(PostTokenizationStep):
    max_length: int
    truncate_right: bool = True
    axis: int = -1

    @classmethod
    def from_hf_json(cls, tokenizer_json: Dict[str, Any], num_of_added_tokens: int = 0) -> "TruncationStep":
        max_length = min(
            tokenizer_json["truncation"]["max_length"] - num_of_added_tokens,
            2**31 - 1 - num_of_added_tokens,
        )
        return cls(
            max_length=max_length,
            truncate_right=tokenizer_json["truncation"]["direction"] == "Right",
        )

    @classmethod
    def from_hf_object(cls, tokenizer: Any, num_of_added_tokens: int = 0) -> "TruncationStep":
        max_length = min(
            tokenizer.model_max_length - num_of_added_tokens,
            2**31 - 1 - num_of_added_tokens,
        )
        return cls(
            max_length=max_length,
            truncate_right=tokenizer.truncation_side == "right",
        )

    @staticmethod
    def validate_inputs(input_nodes):
        if len(input_nodes) != 3:
            raise UserInputError("Only one input ragged tensor is supported as an input for TruncationStep")

    def get_ov_subgraph(self, input_nodes: List[Output]):
        # FIXME: Truncation side (truncate_right) is ignored
        # TODO: Check if axis is the right-most dimension
        self.validate_inputs(input_nodes)

        max_length = opset.minimum(
            opset.subtract(input_nodes[1], input_nodes[0]),
            make_constant_node(self.max_length, Type.i32),
        )
        return [
            input_nodes[0],
            opset.add(input_nodes[0], max_length).output(0),
            input_nodes[2],
        ]


@dataclass
class SpecialTokenWithId:
    token: Optional[str] = None
    _token_id: Optional[int] = None

    def set_token_id(self, vocab: Optional[List[str]]) -> None:
        if vocab is not None and self.token in vocab:
            self._token_id = vocab.index(self.token)


@dataclass
class TokenWithTypeId:
    token_type_id: Optional[int] = None


@dataclass
class AddToken(TokenWithTypeId, SpecialTokenWithId):
    pass


@dataclass
class Sequence(TokenWithTypeId):
    pass


@dataclass
class CombineSegmentsStep(PostTokenizationStep):
    inputs: List[TokenWithTypeId] = field(default_factory=list)
    segment_ids: Optional[List[int]] = None
    axis: int = -1

    def __post_init__(self):
        if self.segment_ids is not None:
            return

        segment_ids_tensor = [node.token_type_id for node in self.inputs]
        if any(segment is None for segment in segment_ids_tensor):
            segment_ids_tensor = [0] * len(self.inputs)

        self.segment_ids = segment_ids_tensor

    def set_tokens_ids(self, vocab: Optional[List[int]]) -> None:
        for input_ in self.inputs:
            if isinstance(input_, AddToken):
                input_.set_token_id(vocab)

    @property
    def number_of_added_tokens(self) -> int:
        return sum(1 for input_ in self.inputs if isinstance(input_, AddToken))

    @classmethod
    def from_hf_json_template_postprocessor(
        cls, tokenizer_json: Dict[str, Any], number_of_inputs: int = 1
    ) -> "CombineSegmentsStep":
        inputs: List[TokenWithTypeId] = []
        if number_of_inputs == 1:
            post_processor = tokenizer_json["post_processor"]["single"]
        else:
            post_processor = tokenizer_json["post_processor"]["pair"]

        for template_dict in post_processor:
            if "SpecialToken" in template_dict:
                step = AddToken(
                    token=template_dict["SpecialToken"]["id"],
                    token_type_id=template_dict["SpecialToken"]["type_id"],
                )
                inputs.append(step)
            else:
                inputs.append(Sequence(token_type_id=template_dict["Sequence"]["type_id"]))
        return cls(inputs)

    @classmethod
    def from_hf_json_bert_postprocessor(
        cls, tokenizer_json: Dict[str, Any], number_of_inputs: int = 1
    ) -> "CombineSegmentsStep":
        post_processor_dict = tokenizer_json["post_processor"]
        inputs: List[TokenWithTypeId] = [
            AddToken(
                token=post_processor_dict["cls"][0],
                token_type_id=0,
            ),
            Sequence(token_type_id=0),
            AddToken(
                token=post_processor_dict["sep"][0],
                token_type_id=0,
            ),
        ]
        if number_of_inputs == 2:
            inputs.extend(
                [
                    Sequence(token_type_id=1),
                    AddToken(
                        token=post_processor_dict["sep"][0],
                        token_type_id=1,
                    ),
                ]
            )
        return cls(inputs)

    @classmethod
    def from_hf_json_roberta_processor(
        cls, tokenizer_json: Dict[str, Any], number_of_inputs: int = 1
    ) -> "CombineSegmentsStep":
        if number_of_inputs == 2:
            raise UserInputError("Two inputs not supported for RoBERTa processor")

        post_processor_dict = tokenizer_json["post_processor"]

        inputs: List[TokenWithTypeId] = [Sequence(token_type_id=0)]

        if not post_processor_dict.get("add_special_tokens", True):
            return cls(inputs)

        inputs.insert(0, AddToken(token=post_processor_dict["cls"][0], token_type_id=0))
        inputs.append(AddToken(token=post_processor_dict["sep"][0], token_type_id=0))
        return cls(inputs)

    def validate_inputs(self, input_nodes: List[Output]) -> None:
        number_of_sequence_inputs = sum(1 for input_ in self.inputs if isinstance(input_, Sequence))
        if number_of_sequence_inputs != len(input_nodes) / 3:
            raise UserInputError(
                f"Number of input nodes: {len(input_nodes)}, must be equal to {number_of_sequence_inputs}"
            )

    def get_ov_subgraph(self, input_nodes):
        self.validate_inputs(input_nodes)

        op_inputs = []
        input_nodes_iter = iter(input_nodes)
        for node in self.inputs:
            if isinstance(node, Sequence):
                op_inputs.extend(islice(input_nodes_iter, 3))
            elif isinstance(node, AddToken):
                # Put a scalar as a ragged tensor with scalar shape and a single element
                op_inputs.extend(make_constant_node(0, Type.i32).outputs())
                op_inputs.extend(make_constant_node(1, Type.i32).outputs())
                op_inputs.append(make_constant_node(np.array([node._token_id]), Type.i32).output(0))
            else:
                raise UserInputError(f"Unexpected node type in CombineSegments: {type(node)}")

        op_inputs.append(make_constant_node(self.segment_ids, Type.i32).output(0))
        return _get_factory().create("CombineSegments", op_inputs).outputs()


@dataclass
class PaddingStep(PostTokenizationStep, SpecialTokenWithId):
    pad_right: bool = True
    token_type_id: Optional[int] = None
    max_length: int = -1
    axis: int = -1

    @classmethod
    def from_hf_json(cls, tokenizer_json: Dict[str, Any]) -> "PaddingStep":
        padding_dict = tokenizer_json["padding"]
        return cls(
            token=padding_dict["pad_token"],
            pad_right=padding_dict["direction"] == "Right",
            token_type_id=padding_dict["pad_type_id"],
            # TODO: Initialize max_length
        )

    @staticmethod
    def validate_inputs(input_nodes: List[Output]) -> None:
        # Suppose input_nodes may have multiple tuples each with 3 tensors represented decomposed ragged tensors
        # We suppose that all ragged tensors represent the same structure and produce the mask only once
        if len(input_nodes) % 3 != 0 or len(input_nodes) < 3:
            raise UserInputError(
                f"Number of input nodes should be divisible by 3 and bigger or equal 3. Got {len(input_nodes)}"
            )

    def get_ov_subgraph(self, input_nodes: List[Output]) -> List[Output]:
        self.validate_inputs(input_nodes)

        outputs = []

        if self.max_length == -1 or self.max_length >= 2**31:
            # Calculate max_length as the maximum ragged length
            max_length = opset.reduce_max(
                opset.subtract(input_nodes[1], input_nodes[0]),
                make_constant_node(0, Type.i32),
            )
        else:
            max_length = make_constant_node(self.max_length, Type.i32)

        names = [TOKEN_IDS_INPUT_NAME, TOKEN_TYPE_IDS_INPUT_NAME][: len(input_nodes) // 3]
        for i, name in enumerate(names):
            cur_outputs = (
                _get_factory()
                .create(
                    "RaggedToDense",
                    input_nodes[3 * i : 3 * (i + 1)]
                    + max_length.outputs()
                    + make_constant_node(0, Type.i32).outputs(),
                )
                .outputs()
            )
            cur_outputs[0].tensor.add_names({name})

            outputs.append(cur_outputs[0])
            if i == 0:
                mask = opset.convert(cur_outputs[1], "i32").output(
                    0
                )  # TODO: Change RaggedToDense to generate mask of any type

        mask.tensor.add_names({ATTENTION_MASK_INPUT_NAME})
        outputs.append(mask)

        return outputs


@dataclass
class DecodingStep(BasePipelineStep):
    pass


@dataclass
class VocabDecoderStep(DecodingStep):
    skip_tokens: Optional[List[int]] = None

    def __post_init__(self):
        if self.skip_tokens is None:
            self.skip_tokens = self.get_pipeline().skip_tokens or {}

    def get_vocab_node_outputs(self) -> Optional[List[Output]]:
        return self.get_pipeline().vocab_node_outputs

    def get_ov_subgraph(self, input_nodes: List[Output]) -> List[Output]:
        input_nodes.extend(self.get_vocab_node_outputs())
        return _get_factory().create("VocabDecoder", input_nodes, {"skip_tokens": self.skip_tokens}).outputs()


@dataclass
class CharsToBytesStep(DecodingStep):
    def get_ov_subgraph(self, input_nodes: List[Output]) -> List[Output]:
        return _get_factory().create("CharsToBytes", input_nodes, {}).outputs()


@dataclass
class RegexDecodingStep(DecodingStep):
    regex_search_pattern: str
    replace_term: str

    @classmethod
    def clean_up_tokenization_spaces(cls) -> "RegexDecodingStep":
        return cls(
            regex_search_pattern=r" ([\\.\\?\\!,])| ('[ms])| (') | ('[rv]e)| (n't)",
            replace_term=r"\1",
        )

    @classmethod
    def replace_end_of_word_suffix(cls, suffix: str = "</w>") -> "RegexDecodingStep":
        return cls(
            regex_search_pattern=suffix,
            replace_term=" ",
        )

    @classmethod
    def replace_continuing_subword_prefix(cls, prefix: str = "##") -> "RegexDecodingStep":
        return cls(
            regex_search_pattern=prefix,
            replace_term="",
        )

    def get_ov_subgraph(self, input_nodes: List[Output]) -> List[Output]:
        input_nodes.extend(
            (
                *self.create_string_constant_node(self.regex_search_pattern).outputs(),
                *self.create_string_constant_node(self.replace_term).outputs(),
            )
        )
        return _get_factory().create("RegexNormalization", input_nodes).outputs()

    @classmethod
    def replace_sp_spaces(cls) -> "RegexDecodingStep":
        return cls(
            regex_search_pattern="▁",
            replace_term=" ",
        )


@dataclass
class TokenizerPipeline:
    steps: List[BasePipelineStep] = field(default_factory=list)
    vocab: Optional[List[str]] = field(default=None, repr=False)
    skip_tokens: Optional[List[int]] = field(default=None, repr=False)
    number_of_inputs: int = 1
    vocab_node_outputs: Optional[List[Output]] = field(default=None, repr=False)

    def get_config(self) -> Dict[str, Dict[str, Any]]:
        return {type(step).__name__: step.get_config() for step in self.steps}

    @singledispatchmethod
    def add_steps(self, steps: Any) -> None:
        raise OVTypeError(f"Type {type(steps)} is not supported")

    @add_steps.register
    def _(self, steps: BasePipelineStep) -> None:
        self.steps.append(steps)
        steps.set_pipeline(self)

    @add_steps.register
    def _(self, steps: list) -> None:
        for step in steps:
            self.steps.append(step)
            step.set_pipeline(self)

    def __getitem__(self, item: int) -> BasePipelineStep:
        return self.steps[item]

    def get_tokenizer_ov_subgraph(self) -> Model:
        string_inputs = [op.Parameter(Type.string, PartialShape(["?"])) for _ in range(self.number_of_inputs)]

        processing_outputs = []
        for input_node in string_inputs:
            input_node = _get_factory().create("StringTensorUnpack", input_node.outputs()).outputs()
            for step in self.normalization_steps:
                input_node = step.get_ov_subgraph(input_node)
            input_node = self.add_ragged_dimension(input_node)

            for step in chain(self.pre_tokenization_steps, self.tokenization_steps):
                input_node = step.get_ov_subgraph(input_node)

            processing_outputs.extend(input_node)

        for step in self.post_tokenization_steps:
            processing_outputs = step.get_ov_subgraph(processing_outputs)

        return Model(processing_outputs, string_inputs, name=TOKENIZER_NAME)

    @property
    def normalization_steps(self) -> List[NormalizationStep]:
        return [step for step in self.steps if isinstance(step, NormalizationStep)]

    @property
    def pre_tokenization_steps(self) -> List[PreTokenizatinStep]:
        return [step for step in self.steps if isinstance(step, PreTokenizatinStep)]

    @property
    def tokenization_steps(self) -> List[TokenizationModelStep]:
        return [step for step in self.steps if isinstance(step, TokenizationModelStep)]

    @property
    def post_tokenization_steps(self) -> List[PostTokenizationStep]:
        return [step for step in self.steps if isinstance(step, PostTokenizationStep)]

    @property
    def decoding_steps(self) -> List[DecodingStep]:
        return [step for step in self.steps if isinstance(step, DecodingStep)]

    @staticmethod
    def add_ragged_dimension(input_node: List[Output]) -> List[Output]:
        shape = opset.shape_of(input_node[0])
        batch_size = opset.gather(shape, as_node(0), as_node(0))
        ragged_begins = opset.range(as_node(0), batch_size, as_node(1), output_type="i32").outputs()
        ragged_ends = opset.range(
            as_node(1), opset.add(batch_size, make_constant_node(1, Type.i64)), as_node(1), output_type="i32"
        ).outputs()
        return ragged_begins + ragged_ends + input_node

    def create_decoding_pipeline(self, input_nodes: List[Output]) -> List[Output]:
        for step in self.decoding_steps:
            pipeline_step = step.get_ov_subgraph(input_nodes)
            input_nodes = pipeline_step

        return _get_factory().create("StringTensorPack", input_nodes).outputs()

    def get_detokenizer_ov_subgraph(self) -> Model:
        if not any(isinstance(step, VocabDecoderStep) for step in self.decoding_steps):
            raise NotImplementedError("Detokenizer is not supported for this model yet!")

        input_node = op.Parameter(Type.i32, PartialShape(["?", "?"]))
        token_ids = input_node
        outputs = self.create_decoding_pipeline([token_ids])
        model = Model(outputs, [input_node], name=DETOKENIZER_NAME)
        model.output().tensor.add_names({STRING_OUTPUT_NAME})
        return model

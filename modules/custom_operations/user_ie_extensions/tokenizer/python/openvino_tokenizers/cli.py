# -*- coding: utf-8 -*-
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from argparse import ArgumentParser, Action
from pathlib import Path

from openvino import save_model, Type

from openvino_tokenizers import convert_tokenizer


class StringToTypeAction(Action):
    string_to_type_dict = {
        "i32": Type.i32,
        "i64": Type.i64,
    }

    def __call__(self, parser, namespace, values, option_string=None) -> None:
        setattr(namespace, self.dest, self.string_to_type_dict[values])


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(
        prog="convert_tokenizer",
        description="Converts tokenizers from Huggingface Hub to OpenVINO Tokenizer model.",
    )
    parser.add_argument(
        "name",
        type=str,
        help=(
            "The model id of a tokenizer hosted inside a model repo on huggingface.co "
            "or a path to a saved Huggingface tokenizer directory"
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path(),
        required=False,
        help="Output directory",
    )
    parser.add_argument(
        "--with-detokenizer",
        required=False,
        action="store_true",
        help="Add a detokenizer model to the output",
    )
    parser.add_argument(
        "--skip_special_tokens",
        required=False,
        action="store_true",
        help=(
            "Produce detokenizer that will skip special tokens during decoding, similar to "
            "huggingface_tokenizer.decode(token_ids, skip_special_tokens=True)."
        ),
    )
    parser.add_argument(
        "--use-fast-false",
        required=False,
        action="store_false",
        help=(
            "Pass `use_fast=False` to `AutoTokenizer.from_pretrained`. It will initialize legacy HuggingFace "
            "tokenizer and then converts it to OpenVINO. Might result in slightly different tokenizer. "
            "See models with _slow suffix https://github.com/openvinotoolkit/openvino_contrib/tree/master/modules/"
            "custom_operations/user_ie_extensions/tokenizer/python#coverage-by-model-type to check the potential "
            "difference between original and OpenVINO tokenizers"
        ),
    )
    parser.add_argument(
        "--trust-remote-code",
        required=False,
        action="store_true",
        help=(
            "Pass `trust_remote_code=True` to `AutoTokenizer.from_pretrained`. It will "
            "execute code present on the Hub on your local machine"
        ),
    )
    parser.add_argument(
        "--tokenizer-output-type",
        required=False,
        action=StringToTypeAction,
        default=Type.i64,
        choices=["i32", "i64"],
        help="Type of the output tensors for tokenizer",
    )
    parser.add_argument(
        "--detokenizer-input-type",
        required=False,
        action=StringToTypeAction,
        default=Type.i64,
        choices=["i32", "i64"],
        help="Type of the input tensor for detokenizer",
    )
    parser.add_argument(
        "--streaming-detokenizer",
        required=False,
        action="store_true",
        help=(
            "[Experimental] Modify SentencePiece based detokenizer to keep spaces leading space. "
            "Can be used to stream a model output without TextStreamer buffer"
        ),
    )
    return parser


def convert_hf_tokenizer() -> None:
    from transformers import AutoTokenizer


    args = get_parser().parse_args()

    print("Loading Huggingface Tokenizer...")
    hf_tokenizer = AutoTokenizer.from_pretrained(args.name, trust_remote_code=args.trust_remote_code)

    print("Converting Huggingface Tokenizer to OpenVINO...")
    converted = convert_tokenizer(
        hf_tokenizer,
        with_detokenizer=args.with_detokenizer,
        tokenizer_output_type=args.tokenizer_output_type,
        detokenizer_input_type=args.detokenizer_input_type,
        streaming_detokenizer=args.streaming_detokenizer,
    )
    if not isinstance(converted, tuple):
        converted = (converted,)

    for converted_model, name in zip(converted, ("tokenizer", "detokenizer")):
        save_path = args.output / f"openvino_{name}.xml"
        save_model(converted_model, save_path)
        print(f"Saved OpenVINO {name.capitalize()}: {save_path}, {save_path.with_suffix('.bin')}")

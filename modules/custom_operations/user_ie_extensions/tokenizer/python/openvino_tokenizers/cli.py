# -*- coding: utf-8 -*-
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from argparse import ArgumentParser
from pathlib import Path

from openvino import save_model

from openvino_tokenizers import convert_tokenizer


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(
        prog="convert_tokenizer", description="Converts tokenizers from Huggingface Hub to OpenVINO Tokenizer model."
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
        "--trust-remote-code",
        required=False,
        action="store_true",
        help=(
            "Pass `trust_remote_code=True` to `AutoTokenizer.from_pretrained`. It will"
            "execute code present on the Hub on your local machine."
        ),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        required=False,
        action="store_true",
    )
    parser.add_argument(
        "-s",
        "--streaming-detokenizer",
        required=False,
        help=(
            "[Experimental] Modify SentencePiece based detokenizer to keep spaces leading space. "
            "Can be used to stream a model output without TextStreamer buffer."
        ),
    )
    return parser


def convert_hf_tokenizer() -> None:
    from transformers import AutoTokenizer

    args = get_parser().parse_args()
    hf_tokenizer = AutoTokenizer.from_pretrained(args.name, trust_remote_code=args.trust_remote_code)
    converted = convert_tokenizer(
        hf_tokenizer, with_detokenizer=args.with_detokenizer, streaming_detokenizer=args.streaming_detokenizer
    )
    if not isinstance(converted, tuple):
        converted = (converted,)

    for converted_model, name in zip(converted, ("tokenizer", "detokenizer")):
        save_model(converted_model, args.output / f"{name}.xml")

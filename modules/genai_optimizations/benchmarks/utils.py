# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


def add_visual_pruning_args(parser):
    group = parser.add_argument_group("Visual Token Pruning Arguments")
    group.add_argument("--enable_visual_pruning", action="store_true", help="Enable visual token pruning")
    group.add_argument("--num_keep_tokens", type=int, default=128, help="Number of visual tokens to keep")
    group.add_argument("--theta", type=float, default=0.5, help="Balance factor for diversity vs relevance")
    return parser


def add_attention_args(parser):
    group = parser.add_argument_group("Attention Kernel Arguments")
    group.add_argument("--use_custom_attention", action="store_true", help="Enable custom attention kernel")
    group.add_argument("--prefill_impl", default="dense", choices=["dense", "tri-shape", "x-attention"])
    group.add_argument("--threshold", type=float, default=0.8, help="Threshold for X-Attention")
    group.add_argument(
        "--last_query_size",
        type=int,
        default=100,
        help="Number of most recent query tokens that attend densely to all previous keys in the Tri-shape pattern"
    )
    group.add_argument(
        "--recent_size",
        type=int,
        default=1024,
        help="Window size of recent tokens each query can attend to in the Tri-shape pattern"
    )
    return parser

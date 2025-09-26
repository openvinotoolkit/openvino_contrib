# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


def add_visual_pruning_args(parser):
    group = parser.add_argument_group("Visual Token Pruning Arguments")
    group.add_argument("--enable_visual_pruning", action="store_true", help="Enable visual token pruning")
    group.add_argument("--num_keep_tokens", type=int, default=128, help="Number of visual tokens to keep")
    group.add_argument("--theta", type=float, default=0.5, help="Balance factor for diversity vs relevance")
    return parser

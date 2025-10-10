# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from genai_opt import SparseAttention
from genai_opt import KVCacheCompressionMode, KVCacheCompressionParameters, KVCacheCompressor

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


def add_token_eviction_args(parser):
    group = parser.add_argument_group("Token Eviction Arguments")
    group.add_argument("--enable_eviction", action="store_true", help="Enable token eviction")
    group.add_argument("--algorithm", default="snapkv", choices=["snapkv", "h2o"], help="The KV cache eviction algorithm")
    group.add_argument("--granularity", default="per_group", choices=["per_token", "per_group"], help="Eviction granularity")
    group.add_argument(
        "--normalize_scores",
        action="store_true",
        help="Whether to normalize the attention scores by the number of times each token was attended to."
    )
    group.add_argument(
        "--start_tokens",
        type=int,
        default=32,
        help="The number of tokens in the beginning of the cache (least recent) to be retained"
    )
    group.add_argument("--intermediate_tokens", type=int, default=1024, help="The number of intermediate tokens to consider for eviction")
    group.add_argument("--recent_tokens", type=int, default=128, help="The number of most recent tokens to be retained")
    group.add_argument("--group_size", type=int, default=32, help="Group size for per-group eviction strategy")
    group.add_argument("--window_size", type=int, default=None, help="The size of the importance score aggregation window")
    return parser


def get_sparse_attention_patcher(args):
    print(f"Enable custom attention kernel with {args.prefill_impl} implementation")
    return SparseAttention(
        algorithm=args.prefill_impl,
        threshold=args.threshold,
        recent_size=args.recent_size,
        last_query_size=args.last_query_size,
        output_attentions=args.enable_eviction,  # output attention weights only if eviction is enabled
    )


def get_eviction_patcher(args):
    print(f"Enable token eviction with {args.algorithm} algorithm")
    algorithm = KVCacheCompressionMode(args.algorithm)
    params = KVCacheCompressionParameters(
        algorithm=algorithm,
        granularity=args.granularity,
        group_size=args.group_size,
        start_tokens=args.start_tokens,
        recent_tokens=args.recent_tokens,
        intermediate_tokens=args.intermediate_tokens,
        normalize_scores=args.normalize_scores,
        window_size=args.window_size,
    )
    return KVCacheCompressor(eviction_parameters=params)

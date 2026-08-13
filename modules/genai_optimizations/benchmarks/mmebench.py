# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
from argparse import ArgumentParser
from contextlib import ExitStack

import torch
from datasets import load_dataset
from genai_opt import get_inputs_embeds
from mme_metrics import calculate_accuracy_plus, calculate_metrics, parse_yes_no
from tqdm import tqdm
from transformers import AutoProcessor, set_seed
from utils import (
    add_attention_args,
    add_token_eviction_args,
    add_visual_pruning_args,
    get_eviction_patcher,
    get_sparse_attention_patcher,
)


@torch.no_grad()
def evaluate(args):
    model_name = args.model
    category = args.subset
    dataset = load_dataset("darkyarding/MME", split="test")
    dataset = dataset.filter(lambda x: x["category"] == category)
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model_cls = get_model_class(model_name)

    kwargs = {"temperature": None, "top_p": None, "top_k": None}
    # force attn_implementation="eager" when using token eviction without custom attention
    if args.enable_eviction and not args.use_custom_attention:
        kwargs["attn_implementation"] = "eager"

    model = model_cls.from_pretrained(
        model_name,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map="auto",
        token=os.environ.get("HF_TOKEN", None),
        **kwargs,
    ).eval()

    if args.enable_visual_pruning:
        print(
            f"Enable visual token pruning with num_keep_tokens={args.num_keep_tokens}, theta={args.theta}"
        )
        num_keep_tokens = args.num_keep_tokens
        theta = args.theta
    else:
        num_keep_tokens = None
        theta = None

    contexts = []
    if args.use_custom_attention:
        sparse_prefill = get_sparse_attention_patcher(args)
        contexts.append(sparse_prefill)

    if args.enable_eviction:
        token_eviction = get_eviction_patcher(args)
        contexts.append(token_eviction)

    all_items = []
    with ExitStack() as stack:
        for ctx in contexts:
            if ctx is not None:
                stack.enter_context(ctx(model))

        for example in tqdm(dataset):
            prompt = example["question"]
            answer = example["answer"].strip().lower()
            image = example["image"].convert("RGB")

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image", "image": image},
                    ],
                }
            ]
            prompt = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = processor(text=prompt, images=image, return_tensors="pt").to(
                model.device
            )

            image_embeds = get_inputs_embeds(
                model, inputs, num_keep_tokens=num_keep_tokens, theta=theta
            )
            kwargs = {}
            if "image_sizes" in inputs:
                kwargs["image_sizes"] = inputs.image_sizes

            generate_ids = model.generate(
                inputs_embeds=image_embeds,
                max_new_tokens=512,
                do_sample=False,
                **kwargs,
            )

            response = processor.batch_decode(
                generate_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            pred_label = parse_yes_no(response)
            all_items.append((image, prompt, answer, pred_label))

    flat_gts = [item[2] for item in all_items]
    flat_preds = [item[3] for item in all_items]
    metrics = calculate_metrics(flat_gts, flat_preds)
    metrics["acc_plus"] = calculate_accuracy_plus(flat_gts, flat_preds)

    print(f"\n MME Evaluation for '{category}'")
    for k, v in metrics.items():
        print(f"{k:>12}: {v:.4f}" if isinstance(v, float) else f"{k:>12}: {v}")


def get_model_class(model_name):
    if "Qwen2.5-VL" in model_name:
        from transformers import Qwen2_5_VLForConditionalGeneration

        return Qwen2_5_VLForConditionalGeneration
    elif "Qwen2-VL" in model_name:
        from transformers import Qwen2VLForConditionalGeneration

        return Qwen2VLForConditionalGeneration
    elif "llava-1.5" in model_name:
        from transformers import LlavaForConditionalGeneration

        return LlavaForConditionalGeneration
    elif "llava-v1.6" in model_name:
        from transformers import LlavaNextForConditionalGeneration

        return LlavaNextForConditionalGeneration
    else:
        error_msg = f"Unsupported model class for: {model_name}"
        raise ValueError(error_msg)


if __name__ == "__main__":
    set_seed(42)

    eval_type_dict = [
        "existence",
        "count",
        "position",
        "color",
        "posters",
        "celebrity",
        "scene",
        "landmark",
        "artwork",
        "OCR",
    ] + [
        "commonsense_reasoning",
        "numerical_calculation",
        "text_translation",
        "code_reasoning",
    ]

    parser = ArgumentParser()
    parser.add_argument(
        "--model", type=str, required=True, help="Huggingface model repo"
    )
    parser.add_argument(
        "--subset", choices=eval_type_dict, required=True, help="MME category name"
    )

    add_visual_pruning_args(parser)
    add_attention_args(parser)
    add_token_eviction_args(parser)
    args = parser.parse_args()

    evaluate(args)

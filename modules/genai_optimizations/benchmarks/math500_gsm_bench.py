# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# This logic is largely copied from the
# - https://github.com/microsoft/ProphetNet/tree/master/CRITIC
# - https://github.com/openai/prm800k
# - https://github.com/microsoft/ToRA/blob/main/src/eval/grader.py
# - https://github.com/deepseek-ai/DeepSeek-Math/blob/main/evaluation/eval/eval_utils.py
# - https://github.com/VITA-Group/SEAL/tree/main

import argparse
import json
import os
import random
import re
from collections import Counter
from contextlib import ExitStack

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

from utils import add_attention_args, add_token_eviction_args
from utils import get_eviction_patcher, get_sparse_attention_patcher

from reasoning_parser import extract_answer
from reasoning_parser import parallel_math_equal
from reasoning_parser import strip_string

# disable tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
OUTPUT_LENGTHS = []


def run_evaluation(res_path, save=False, k=None, output_dir=None):
    with open(res_path) as f:
        lines = f.readlines()
        data = [json.loads(line) for line in lines]

    for example in tqdm(data):
        if "model_generation" not in example:
            example["model_generation"] = example["model_output"]
        if k is not None:
            example["model_generation"] = example["model_generation"][:k]
        gt_cot = example["answer"]
        gt_ans = extract_answer(gt_cot, data_name="omni-math")
        gt_cot = str(gt_cot).strip()
        gt_ans = strip_string(gt_ans, skip_unit=False)
        all_pred = [extract_answer(p, data_name="omni-math") for p in example["model_generation"]]
        all_pred = [strip_string(p, skip_unit=False) for p in all_pred]
        all_eval = parallel_math_equal(all_pred, gt_ans, timeout=5)
        effective_pred = [p for p, o in zip(all_pred, example["model_generation"]) if "boxed" in o]
        if len(effective_pred) == 0:
            effective_pred = all_pred
        counter = Counter(effective_pred)
        pred = counter.most_common(1)[0][0]
        index = all_pred.index(pred)
        eval = all_eval[index]
        example["all_pred"] = all_pred
        example["all_eval"] = all_eval
        example["mv_pred"] = pred
        example["mv_eval"] = eval
        example["mv_index"] = index

    acc = sum([example["mv_eval"] for example in data]) / len(data)
    print(f"Accuracy: {acc:.3f}")

    correct_avg_len = []
    incorrect_avg_len = []

    for i, example in enumerate(data):
        if example["mv_eval"]:
            correct_avg_len.append(OUTPUT_LENGTHS[i])
        else:
            incorrect_avg_len.append(OUTPUT_LENGTHS[i])

    if len(correct_avg_len) != 0:
        print(f"Correct avg len: {sum(correct_avg_len) / len(correct_avg_len):.2f}", end=", ")
    if len(incorrect_avg_len) != 0:
        print(f"Incorrect avg len: {sum(incorrect_avg_len) / len(incorrect_avg_len):.2f}")

    if save:
        out_file = os.path.join(output_dir, "math_eval.jsonl")
        with open(out_file, "w") as f:
            for example in data:
                f.write(json.dumps(example) + "\n")

        metric_file = os.path.join(output_dir, "metrics.json")
        with open(metric_file, "w") as f:
            json.dump({"acc": acc}, f)


def trim_output(output):
    instruction_prefix = "Answer the following question"
    question_prefix = "Question:"
    comment_prefix = "Comment:"  # for some reason, Llama 13B likes to generate these comments indefinitely

    for prefix in [instruction_prefix, question_prefix, comment_prefix]:
        if prefix in output:
            output = output.split(prefix)[0]

    return output


def extract_box(pred_str):
    ans = pred_str.split("boxed")[-1]
    if len(ans) == 0:
        return ""
    elif ans[0] == "{":
        stack = 1
        a = ""
        for c in ans[1:]:
            if c == "{":
                stack += 1
                a += c
            elif c == "}":
                stack -= 1
                if stack == 0:
                    break
                a += c
            else:
                a += c
    else:
        a = ans.split("$")[0].strip()

    return a


def prepare_dataset(dataset, max_samples=None):
    test_data = []
    if dataset == "MATH500":
        data = load_dataset("HuggingFaceH4/MATH-500", split="test")
        for example in data:
            gt = extract_box(example["solution"])
            test_data.append(
                {
                    "question": example["problem"],
                    "answer": example["solution"],
                    "gt": gt,
                }
            )
    elif dataset == "GSM":
        data_path = "gsm.jsonl"

        if not os.path.exists(data_path):
            import requests
            url = "https://raw.githubusercontent.com/VITA-Group/SEAL/main/data/gsm/test.jsonl"
            response = requests.get(url)
            response.raise_for_status()
            with open(data_path, "w", encoding="utf-8") as f:
                f.write(response.text)
            print(f"Downloaded and saved to '{data_path}'.")

        with open(data_path) as fin:
            for line in fin:
                example = json.loads(line)
                answer = example["answer"].split("####")[1].strip()
                answer = re.sub(r"(\d),(\d)", r"\1\2", answer)
                test_data.append(
                    {
                        "question": example["question"],
                        "answer": example["answer"].split("####")[0].strip(),
                        "gt": answer,
                    }
                )

    if max_samples and len(test_data) > max_samples:
        test_data = test_data[:max_samples]

    return test_data


def main(args):
    random.seed(42)

    test_data = prepare_dataset(args.dataset, max_samples=args.max_examples)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # set pad token to eos token if pad token is not set (as is the case for llama models)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    prefix = (
        "Answer the following questions. You should think step-by-step and put your final answer within \\boxed{}.\n"
    )
    prompts = []
    for example in test_data:
        prompt = prefix + "Question: " + example["question"].strip() + "\nAnswer: "
        if not args.omit_chat_template:
            if "deepseek" in args.model:
                messages = [{"role": "user", "content": prefix + "Question: " + example["question"].strip()}]
            else:
                messages = [
                    {"role": "system", "content": prefix},
                    {"role": "user", "content": "Question: " + example["question"].strip()},
                ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            if not args.keep_bos and tokenizer.bos_token is not None and prompt.startswith(tokenizer.bos_token):
                prompt = prompt[len(tokenizer.bos_token) :]
        prompts.append(prompt)

    kwargs = {"temperature": None, "top_p": None, "top_k": None}
    # force attn_implementation="eager" when using token eviction without custom attention
    if args.enable_eviction and not args.use_custom_attention:
        kwargs["attn_implementation"] = "eager"

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        device_map="auto",
        token=os.environ.get("HF_TOKEN", None),
        **kwargs
    )
    model.eval()

    contexts = []
    if args.use_custom_attention:
        sparse_attn = get_sparse_attention_patcher(args)
        contexts.append(sparse_attn)

    if args.enable_eviction:
        token_eviction = get_eviction_patcher(args)
        contexts.append(token_eviction)

    outputs = []
    avg_prompt_len = []
    with ExitStack() as stack:
        for ctx in contexts:
            if ctx is not None:
                stack.enter_context(ctx(model))

        for prompt in tqdm(prompts):
            tokenized_batch = tokenizer(prompt, return_tensors="pt", padding=True)
            tokenized_batch = {k: v.to(model.device) for k, v in tokenized_batch.items()}
            avg_prompt_len.append(tokenized_batch["input_ids"].shape[1])

            output = model.generate(
                **tokenized_batch,
                do_sample=False,
                max_new_tokens=args.max_tokens,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
            )
            OUTPUT_LENGTHS.append(output.shape[1])
            output = [tokenizer.decode(o[avg_prompt_len[-1]:], skip_special_tokens=True) for o in output]
            outputs.extend(output)

    outputs = [[trim_output(o)] for o in outputs]
    print(f"Average prompt length: {sum(avg_prompt_len) / len(avg_prompt_len):.2f}")
    print(f"Average length: {sum(OUTPUT_LENGTHS) / len(OUTPUT_LENGTHS):.2f}")

    predictions = [
        {
            "prompt": prompt,
            "problem": example["question"],
            "answer": example["gt"],
            "solution": example["answer"],
            "model_generation": output,
        }
        for example, output, prompt in zip(test_data, outputs, prompts)
    ]

    with open(os.path.join(args.save_dir, "predictions.jsonl"), "w") as fout:
        for prediction in predictions:
            fout.write(json.dumps(prediction) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="MATH500", choices=["MATH500", "GSM"])
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--save_dir", type=str, default="results")
    parser.add_argument("--max_tokens", type=int, default=5000)
    parser.add_argument("--omit_chat_template", action="store_true")
    parser.add_argument("--keep_bos", action="store_true")

    add_attention_args(parser)
    add_token_eviction_args(parser)
    args = parser.parse_args()

    args.save_dir = os.path.join(args.save_dir, args.dataset)
    if args.keep_bos:
        args.save_dir = args.save_dir + "_keep_bos"

    if args.max_examples or args.start:
        start = 0 if args.start is None else args.start
        end = start + args.max_examples if args.max_examples is not None else -1
        args.save_dir = os.path.join(args.save_dir, f"{start}_{end}")

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    print(f"Results will be saved to {args.save_dir}")
    main(args)
    run_evaluation(os.path.join(args.save_dir, "predictions.jsonl"), output_dir=args.save_dir)

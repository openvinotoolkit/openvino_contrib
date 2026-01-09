# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# This logic is largely copied from the https://github.com/THUDM/LongBench
import gc
import os
import re
import string
from argparse import ArgumentParser
from collections import Counter
from contextlib import ExitStack

import datasets
import torch
import transformers
from fuzzywuzzy import fuzz
from rouge import Rouge
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

from utils import add_attention_args, add_token_eviction_args
from utils import get_eviction_patcher, get_sparse_attention_patcher

# (Phi3 and DeepSeek issue)
# AttributeError: 'DynamicCache' object has no attribute 'get_max_length'. Did you mean: 'get_seq_length'?
# The method get_max_length of 'DynamicCache' is deprecated and has been removed in transformer 4.49
# fix: https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite/commit/faea2faa9ec002397d20e90dae777c4252022f3a

class LongBenchDataset:
    def __init__(self, subset, model_name):
        self.subset = subset
        self.model_name = model_name
        self.dataset = datasets.load_dataset("THUDM/LongBench", subset, split="test")

        self.dataset2metric = {
            "narrativeqa": self.qa_f1_score,
            "qasper": self.qa_f1_score,
            "multifieldqa_en": self.qa_f1_score,
            "hotpotqa": self.qa_f1_score,
            "2wikimqa": self.qa_f1_score,
            "musique": self.qa_f1_score,
            "gov_report": self.rouge_score,
            "qmsum": self.rouge_score,
            "multi_news": self.rouge_score,
            "trec": self.classification_score,
            "triviaqa": self.qa_f1_score,
            "samsum": self.rouge_score,
            "lsht": self.classification_score,
            "passage_retrieval_en": self.retrieval_score,
            "passage_count": self.count_score,
            "passage_retrieval_zh": self.retrieval_zh_score,
            "lcc": self.code_sim_score,
            "repobench-p": self.code_sim_score,
        }

        self.dataset2maxlen = {
            "narrativeqa": 128,
            "qasper": 128,
            "multifieldqa_en": 64,
            "multifieldqa_zh": 64,
            "hotpotqa": 32,
            "2wikimqa": 32,
            "musique": 32,
            "dureader": 128,
            "gov_report": 512,
            "qmsum": 512,
            "multi_news": 512,
            "vcsum": 512,
            "trec": 64,
            "triviaqa": 32,
            "samsum": 128,
            "lsht": 64,
            "passage_count": 32,
            "passage_retrieval_en": 32,
            "passage_retrieval_zh": 32,
            "lcc": 64,
            "repobench-p": 64,
        }

        self.dataset2prompt = {
            "narrativeqa": (
                "You are given a story, which can be either a novel or a movie script, and a question. "
                "Answer the question as concisely as you can, using a single phrase if possible. Do not "
                "provide any explanation.\n\nStory: {context}\n\nNow, answer the question based on the story "
                "as concisely as you can, using a single phrase if possible. Do not provide any explanation.\n\n"
                "Question: {input}\n\nAnswer:"
            ),
            "qasper": (
                "You are given a scientific article and a question. Answer the question as concisely as you can, "
                "using a single phrase or sentence if possible. If the question cannot be answered based on the "
                'information in the article, write "unanswerable". If the question is a yes/no question, answer '
                '"yes", "no", or "unanswerable". Do not provide any explanation.\n\nArticle: {context}\n\n Answer '
                "the question based on the above article as concisely as you can, using a single phrase or sentence if "
                'possible. If the question cannot be answered based on the information in the article, write "unanswerable". '
                'If the question is a yes/no question, answer "yes", "no", or "unanswerable". Do not provide any explanation.'
                "\n\nQuestion: {input}\n\nAnswer:"
            ),
            "multifieldqa_en": (
                "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following question based on "
                "the above text, only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:"
            ),
            "multifieldqa_zh": (
                "阅读以下文字并用中文简短回答：\n\n{context}\n\n现在请基于上面的文章回答下面的问题，只告诉我答案，"
                "不要输出任何其他字词。\n\n问题：{input}\n回答："
            ),
            "hotpotqa": (
                "Answer the question based on the given passages. Only give me the answer and do not output any other words."
                "\n\nThe following are given passages.\n{context}\n\n"
                "Answer the question based on the given passages. Only give me "
                "the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:"
            ),
            "2wikimqa": (
                "Answer the question based on the given passages. Only give me the answer and do not output any other words."
                "\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. "
                "Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:"
            ),
            "musique": (
                "Answer the question based on the given passages. Only give me the answer and do not output any other words."
                "\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. "
                "Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:"
            ),
            "dureader": (
                "请基于给定的文章回答下述问题。\n\n文章：{context}\n\n请基于上述文章回答下面的问题。\n\n问题：{input}\n回答："
            ),
            "gov_report": (
                "You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:"
                "\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:"
            ),
            "qmsum": (
                "You are given a meeting transcript and a query containing a question or instruction. Answer the query "
                "in one or more sentences.\n\nTranscript:\n{context}\n\nNow, answer the query "
                "based on the above meeting transcript in one or more sentences.\n\nQuery: {input}\nAnswer:"
            ),
            "multi_news": (
                "You are given several news passages. Write a one-page summary of all news.\n\nNews:\n{context}\n\n"
                "Now, write a one-page summary of all the news.\n\nSummary:"
            ),
            "vcsum": ("下面有一段会议记录，请你阅读后，写一段总结，总结会议的内容。\n会议记录：\n{context}\n\n会议总结："),
            "trec": (
                "Please determine the type of the question below. Here are some examples of questions.\n\n{context}\n{input}"
            ),
            "triviaqa": (
                "Answer the question based on the given passage. Only give me the answer and do not output any other words. "
                "The following are some examples.\n\n{context}\n\n{input}"
            ),
            "samsum": (
                "Summarize the dialogue into a few short sentences. The following are some examples.\n\n{context}\n\n{input}"
            ),
            "lsht": ("请判断给定新闻的类别，下面是一些例子。\n\n{context}\n{input}"),
            "passage_count": (
                "There are some paragraphs below sourced from Wikipedia. Some of them may be duplicates. Please carefully "
                "read these paragraphs and determine how many unique paragraphs there are after removing duplicates. In other "
                "words, how many non-repeating paragraphs are there in total?\n\n{context}\n\nPlease enter the final count of "
                "unique paragraphs after removing duplicates. The output format should only contain the number, "
                "such as 1, 2, 3, and so on.\n\nThe final answer is: "
            ),
            "passage_retrieval_en": (
                "Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the abstract "
                "is from.\n\n{context}\n\nThe following is an abstract.\n\n{input}\n\nPlease enter the number of the paragraph"
                ' that the abstract is from. The answer format must be like "Paragraph 1", "Paragraph 2", etc.\n\n'
                "The answer is:"
            ),
            "passage_retrieval_zh": (
                "以下是若干段落文字，以及其中一个段落的摘要。请确定给定的摘要出自哪一段。\n\n{context}\n\n下面是一个摘要\n\n"
                '{input}\n\n请输入摘要所属段落的编号。答案格式必须是"段落1"，"段落2"等格式\n\n答案是：'
            ),
            "lcc": "Please complete the code given below. \n{context}Next line of code:\n",
            "repobench-p": "Please complete the code given below. \n{context}{input}Next line of code:\n",
        }

    def get_max_new_tokens(self):
        return self.dataset2maxlen[args.subset]

    def preprocess_prompt(self, data_sample):
        prompt_format = self.dataset2prompt[self.subset]
        prompt = prompt_format.format(**data_sample)
        return prompt

    def post_process_pred(self, pred):
        if self.subset in ["samsum", "qsum", "hotpotqa", "qasper"] and "Llama-3-" in self.model_name:
            pred = pred[: pred.find("assistant")]
        elif self.subset == "samsum":
            pred = pred[: pred.find("\nDialogue")]
        elif "Phi-3" in self.model_name and self.subset == "hotpotqa":
            pred = pred.lstrip("\n").split("\n")[0]
        elif self.subset in ["trec", "hotpotqa", "qasper"]:
            pred = pred[: pred.find("\nQuestion")]
        return pred

    def __iter__(self):
        for item in self.dataset:
            yield item

    def normalize_answer(self, s):
        def remove_articles(text):
            return re.sub(r"\b(a|an|the)\b", " ", text)

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def normalize_zh_answer(self, s):
        def white_space_fix(text):
            return "".join(text.split())

        def remove_punc(text):
            cn_punctuation = (
                "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗"
                "〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
            )
            all_punctuation = set(string.punctuation + cn_punctuation)
            return "".join(ch for ch in text if ch not in all_punctuation)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_punc(lower(s)))

    def count_score(self, prediction, ground_truth, **kwargs):
        numbers = re.findall(r"\d+", prediction)
        right_num = sum(1 for n in numbers if str(n) == str(ground_truth))
        return float(0.0 if not numbers else right_num / len(numbers))

    def retrieval_score(self, prediction, ground_truth, **kwargs):
        matches = re.findall(r"Paragraph (\d+)", ground_truth)
        ground_truth_id = matches[0]
        numbers = re.findall(r"\d+", prediction)
        right_num = sum(1 for n in numbers if str(n) == str(ground_truth_id))
        return float(0.0 if not numbers else right_num / len(numbers))

    def retrieval_zh_score(self, prediction, ground_truth, **kwargs):
        matches = re.findall(r"段落(\d+)", ground_truth)
        ground_truth_id = matches[0]
        numbers = re.findall(r"\d+", prediction)
        right_num = sum(1 for n in numbers if str(n) == str(ground_truth_id))
        return float(0.0 if not numbers else right_num / len(numbers))

    def code_sim_score(self, prediction, ground_truth, **kwargs):
        all_lines = prediction.lstrip("\n").split("\n")
        prediction = ""
        for line in all_lines:
            if ("`" not in line) and ("#" not in line) and ("//" not in line):
                prediction = line
                break
        return fuzz.ratio(prediction, ground_truth) / 100

    def classification_score(self, prediction, ground_truth, **kwargs):
        all_classes = kwargs["all_classes"]
        matches = [c for c in all_classes if c in prediction]
        # Filter partial matches
        matches = [
            m for m in matches
            if not (m in ground_truth and m != ground_truth)
        ]
        return 1.0 / len(matches) if ground_truth in matches else 0.0

    def rouge_score(self, prediction, ground_truth, **kwargs):
        rouge = Rouge()
        try:
            scores = rouge.get_scores([prediction], [ground_truth], avg=True)
        except Exception:
            return 0.0
        return scores["rouge-l"]["f"]

    def f1_score(self, prediction, ground_truth, **kwargs):
        common = Counter(prediction) & Counter(ground_truth)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = num_same / len(prediction)
        recall = num_same / len(ground_truth)
        return (2 * precision * recall) / (precision + recall)

    def qa_f1_score(self, prediction, ground_truth, **kwargs):
        p = self.normalize_answer(prediction)
        g = self.normalize_answer(ground_truth)
        return self.f1_score(p.split(), g.split())

    def scorer(self, predictions, answers, all_classes):
        total_score = 0.0
        for pred, gts in zip(predictions, answers):
            score = 0.0
            if self.subset in ["trec", "triviaqa", "samsum", "lsht"]:
                pred = pred.lstrip("\n").split("\n")[0]
            for gt in gts:
                func = self.dataset2metric[self.subset]
                score = max(score, func(pred, gt, all_classes=all_classes))
            total_score += score
        return round(100 * total_score / len(predictions), 2)

    def get_score(self, model_output):
        predictions, answers = [], []
        all_classes = None
        for item in model_output:
            predictions.append(item["pred"])
            answers.append(item["answers"])
            all_classes = item["all_classes"]
        return self.scorer(predictions, answers, all_classes)


@torch.no_grad()
def evaluate(args):
    dataset = LongBenchDataset(args.subset, args.model)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, token=os.environ.get("HF_TOKEN", None)
    )

    kwargs = {"temperature": None, "top_p": None, "top_k": None}
    # force attn_implementation="eager" when using token eviction without custom attention
    if args.enable_eviction and not args.use_custom_attention:
        kwargs["attn_implementation"] = "eager"

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        dtype=torch.float16,
        device_map="auto",
        token=os.environ.get("HF_TOKEN", None),
        **kwargs,
    ).eval()

    patchers = []
    if args.use_custom_attention:
        sparse_attn = get_sparse_attention_patcher(args)
        patchers.append(sparse_attn)

    if args.enable_eviction:
        token_eviction = get_eviction_patcher(args)
        patchers.append(token_eviction)

    max_new_tokens = dataset.get_max_new_tokens()
    answers = []
    max_length = 4500

    with ExitStack() as stack:
        for patcher in patchers:
            if patcher is not None:
                stack.enter_context(patcher(model))

        for data_sample in tqdm(dataset):
            prompt = dataset.preprocess_prompt(data_sample)
            inputs = tokenizer([prompt], truncation=False, return_tensors="pt").to(model.device)
            if len(inputs.input_ids[0]) > max_length:
                half = int(max_length / 2)
                prompt = tokenizer.decode(inputs.input_ids[0][:half], skip_special_tokens=True) + tokenizer.decode(
                    inputs.input_ids[0][-half:], skip_special_tokens=True
                )
                inputs = tokenizer([prompt], truncation=False, return_tensors="pt").to(model.device)

            context_length = inputs.input_ids.shape[-1]
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
            )[0]

            generate_answer = tokenizer.decode(outputs[context_length:], skip_special_tokens=True)
            answers.append(
                {
                    "answers": data_sample["answers"],
                    "all_classes": data_sample["all_classes"],
                    "pred": dataset.post_process_pred(generate_answer),
                }
            )
            del inputs, outputs
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            gc.collect()

    score = dataset.get_score(answers)
    print(f"Score: {score}")


if __name__ == "__main__":
    transformers.set_seed(42)

    parser = ArgumentParser()
    parser.add_argument("--subset", type=str, required=True, help="Dataset name")
    parser.add_argument("--model", type=str, required=True, help="Model name")

    add_attention_args(parser)
    add_token_eviction_args(parser)
    args = parser.parse_args()

    evaluate(args)

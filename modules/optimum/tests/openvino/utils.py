from typing import Optional, Sequence

import numpy as np
import torch
from datasets import load_metric
from optimum.intel.openvino import OVAutoModelForQuestionAnswering
from transformers import AutoModelForQuestionAnswering, AutoTokenizer


class QAModel:
    def __init__(self, model_name, framework, max_seq_length=None):
        """
        Wrapper class for QuestionAnswering models

        :param model_name: model name, e.g. "bert-base-uncased". Should refer to a PyTorch
                           model on Hugging Face Model Hub
        :param framework: "ov" or "pytorch"
        :param max_seq_length: max sequence length for inference. e.g. 128
        """
        if framework == "ov":
            self.model = OVAutoModelForQuestionAnswering.from_pretrained(model_name, from_pt=True)
        else:
            self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_seq_length = max_seq_length

    def ask(self, question, context):
        torch.set_grad_enabled(False)
        encoded_input = self.tokenizer.encode_plus(
            question,
            context,
            return_tensors="pt",
            add_special_tokens=True,
            max_length=512 if self.max_seq_length is None else min(self.max_seq_length, 512),
            padding="max_length" if self.max_seq_length is not None else "do_not_pad",
            truncation=True,
        )
        # Do inference
        result = self.model(**encoded_input, return_dict=True)

        # Convert inference result to answer
        answer_start_scores = np.array(result["start_logits"])
        answer_end_scores = np.array(result["end_logits"])
        answer_start = np.argmax(answer_start_scores)
        answer_end = np.argmax(answer_end_scores) + 1
        # Convert tokens to answer string
        input_ids = encoded_input["input_ids"].tolist()[0]
        answer = self.tokenizer.convert_tokens_to_string(
            self.tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end])
        )
        return answer

    def compute_metrics(self, examples):
        metric = load_metric("squad")
        f1 = []
        em = []
        for item in examples:
            answer = self.ask(item["question"], item["context"])
            references = [{"id": item["id"], "answers": item["answers"]}]
            predictions = [{"id": item["id"], "prediction_text": answer}]
            metric_scores = metric.compute(references=references, predictions=predictions)
            f1.append(metric_scores["f1"])
            em.append(metric_scores["exact_match"])
        return np.mean(f1), np.mean(em)


def generate_random_inputs(
    num_items: int,
    input_names: Sequence,
    seq_length: int,
    std: Optional[int] = None,
    seed: int = 141,
    return_tensors="np",
):
    """
    Return list of input data dictionaries with `num_items` items. Sequences have length `seq_length`. If
    std is given, sequences of a random distribution around length will be returned, otherwise fixed length
    sequences of seq_length will be returned.

    :param num_items: number of input data dictionaries to return
    :param input_names: model input names. Example ["input_ids", "attention_mask"]
    :param seq_length: sequence length. Example: 128
    :param std: standard deviation for sequence length. If given, return input_data with dynamic shapes with
        mean `seq_length`. If None, return input_data with static shapes of `seq_length`
    :param seed: random seed for reproducibility
    :param return_tensors: "np" for Numpy tensors, "pt" for PyTorch tensors
    :return: list with `num_items` dictionaries with input data,
        where dictionaries have keys with `input_names`
    """

    if return_tensors == "pt":
        import torch

        random_func = torch.randint
    else:
        random_func = np.random.randint

    assert len(input_names) >= 2
    if std is not None:
        np.random.seed(seed)
        clip_length = 512  # hardcoded max model input shape
        seq_lengths = np.random.normal(seq_length, std, num_items).clip(0, clip_length)
        seq_lengths = (seq_lengths - seq_lengths.mean()) / seq_lengths.std()
        seq_lengths = seq_lengths * std + seq_length
        seq_lengths = seq_lengths.clip(0, clip_length).round().astype(np.uint16)
    else:
        seq_lengths = np.repeat(seq_length, num_items)

    input_list = []
    for generated_seq_length in seq_lengths:
        input_data = {}
        for input_name in input_names:
            high = 10000 if input_name == "input_ids" else 1
            input_data[input_name] = random_func(low=0, high=high, size=(1, generated_seq_length))
        input_list.append(input_data)
    return input_list

# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import re
from collections.abc import Iterable, Sequence

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)

VALID_ANSWERS = frozenset({"yes", "no"})


def parse_yes_no(text: str) -> str:
    """Return a leading yes/no answer in a response, or ``other``."""
    match = re.match(
        r"\s*[\W_]*(?:(?:the\s+)?answer\s*(?:is\s*)?[:=-]?\s*[\W_]*)?"
        r"(yes|no)\b",
        text,
        flags=re.IGNORECASE,
    )
    return match.group(1).lower() if match else "other"


def iter_pairs(items: Sequence, pair_size: int = 2) -> Iterable[Sequence]:
    if pair_size <= 0:
        raise ValueError("pair_size must be greater than zero")
    for start in range(0, len(items), pair_size):
        yield items[start : start + pair_size]


def calculate_metrics(
    ground_truth: Sequence[str], predictions: Sequence[str]
) -> dict[str, float | int]:
    """Calculate the MME binary metrics with scikit-learn."""
    if len(ground_truth) != len(predictions):
        raise ValueError("ground_truth and predictions must have the same length")
    if not ground_truth:
        raise ValueError("at least one prediction is required")

    invalid_ground_truth = set(ground_truth) - VALID_ANSWERS
    if invalid_ground_truth:
        raise ValueError(
            f"unsupported ground-truth labels: {sorted(invalid_ground_truth)}"
        )

    valid_indices = [
        index
        for index, prediction in enumerate(predictions)
        if prediction in VALID_ANSWERS
    ]
    clean_ground_truth = [ground_truth[index] for index in valid_indices]
    clean_predictions = [predictions[index] for index in valid_indices]
    if clean_ground_truth:
        matrix = confusion_matrix(
            clean_ground_truth,
            clean_predictions,
            labels=["yes", "no"],
        )
        true_positive, false_negative = matrix[0]
        false_positive, true_negative = matrix[1]
        precision = precision_score(
            clean_ground_truth,
            clean_predictions,
            pos_label="yes",
            zero_division=0,
        )
        recall = recall_score(
            clean_ground_truth,
            clean_predictions,
            pos_label="yes",
            zero_division=0,
        )
    else:
        true_positive = false_negative = false_positive = true_negative = 0
        precision = recall = 0.0

    return {
        "TP": int(true_positive),
        "FN": int(false_negative),
        "TN": int(true_negative),
        "FP": int(false_positive),
        "precision": float(precision),
        "recall": float(recall),
        "other_num": len(predictions) - len(valid_indices),
        "acc": float(accuracy_score(ground_truth, predictions)),
    }


def calculate_accuracy_plus(
    ground_truth: Sequence[str], predictions: Sequence[str], pair_size: int = 2
) -> float:
    """Return the fraction of complete image groups answered correctly."""
    if len(ground_truth) != len(predictions):
        raise ValueError("ground_truth and predictions must have the same length")
    groups = list(
        iter_pairs(list(zip(ground_truth, predictions, strict=True)), pair_size)
    )
    complete_groups = [group for group in groups if len(group) == pair_size]
    if not complete_groups:
        return 0.0
    correct_groups = sum(
        all(expected == predicted for expected, predicted in group)
        for group in complete_groups
    )
    return correct_groups / len(complete_groups)

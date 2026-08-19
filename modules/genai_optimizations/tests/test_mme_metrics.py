# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

pytest.importorskip("sklearn")

from mme_metrics import (
    calculate_accuracy_plus,
    calculate_metrics,
    iter_pairs,
    parse_yes_no,
)


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("Yes", "yes"),
        ("  **NO**, because...", "no"),
        ("Answer: yes", "yes"),
        ("The answer is no", "no"),
        ("**Answer:** Yes, because...", "yes"),
        ("yesterday", "other"),
        ("The response discusses yes and no", "other"),
        ("unknown", "other"),
    ],
)
def test_parse_yes_no(response, expected):
    assert parse_yes_no(response) == expected


def test_calculate_metrics_counts_invalid_predictions_as_incorrect():
    metrics = calculate_metrics(
        ["yes", "yes", "no", "no", "yes"],
        ["yes", "no", "yes", "no", "other"],
    )

    assert metrics == {
        "TP": 1,
        "FN": 1,
        "TN": 1,
        "FP": 1,
        "precision": 0.5,
        "recall": 0.5,
        "other_num": 1,
        "acc": 0.4,
    }


def test_metrics_handle_zero_denominators():
    metrics = calculate_metrics(["no"], ["other"])
    assert metrics["precision"] == 0
    assert metrics["recall"] == 0


def test_accuracy_plus_requires_every_answer_in_a_pair():
    assert (
        calculate_accuracy_plus(
            ["yes", "no", "yes", "no"],
            ["yes", "no", "yes", "yes"],
        )
        == 0.5
    )


def test_pair_validation():
    with pytest.raises(ValueError):
        list(iter_pairs([1, 2], pair_size=0))
    with pytest.raises(ValueError):
        calculate_metrics(["yes"], [])
    with pytest.raises(ValueError):
        calculate_metrics(["maybe"], ["yes"])

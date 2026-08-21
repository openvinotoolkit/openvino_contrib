# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

pytest.importorskip("math_verify")

from math_evaluation import evaluate_math_predictions


def test_evaluate_math_predictions_extracts_and_verifies_answers():
    predictions, results = evaluate_math_predictions(
        r"The answer is \boxed{\frac{1}{2}}.",
        ["0.5", r"The final answer is \boxed{\frac{2}{4}}.", "0.25"],
    )

    assert predictions == ["0.500000000000000", "1/2", "0.250000000000000"]
    assert results == [True, True, False]


def test_evaluate_math_predictions_handles_unparseable_output():
    predictions, results = evaluate_math_predictions("42", ["No numeric answer"])

    assert predictions == [""]
    assert results == [False]

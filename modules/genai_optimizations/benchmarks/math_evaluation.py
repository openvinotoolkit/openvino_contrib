# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence

from math_verify import parse, verify


def evaluate_math_predictions(
    reference: str,
    predictions: Sequence[str],
    timeout_seconds: int = 5,
) -> tuple[list[str], list[bool]]:
    parsed_reference = parse(str(reference), parsing_timeout=timeout_seconds)
    parsed_predictions = [
        parse(str(prediction), parsing_timeout=timeout_seconds)
        for prediction in predictions
    ]
    canonical_predictions = [
        str(parsed_prediction[0]) if parsed_prediction else ""
        for parsed_prediction in parsed_predictions
    ]
    results = [
        verify(
            parsed_reference,
            parsed_prediction,
            timeout_seconds=timeout_seconds,
        )
        for parsed_prediction in parsed_predictions
    ]
    return canonical_predictions, results

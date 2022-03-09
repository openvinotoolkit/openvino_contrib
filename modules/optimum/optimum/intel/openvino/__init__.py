# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .modeling_ov_auto import (
    OVAutoModel,
    OVAutoModelForMaskedLM,
    OVAutoModelWithLMHead,
    OVAutoModelForQuestionAnswering,
    OVAutoModelForSequenceClassification,
    OVAutoModelForAudioClassification,
)

__all__ = [
    "OVAutoModel",
    "OVAutoModelForMaskedLM",
    "OVAutoModelWithLMHead",
    "OVAutoModelForQuestionAnswering",
    "OVAutoModelForSequenceClassification",
    "OVAutoModelForAudioClassification",
]

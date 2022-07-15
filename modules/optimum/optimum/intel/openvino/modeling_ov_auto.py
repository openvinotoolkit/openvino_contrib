# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from transformers.file_utils import is_torch_available, is_tf_available
from .modeling_ov_utils import OVPreTrainedModel


if is_torch_available():
    from transformers import (
        AutoModel,
        AutoModelForQuestionAnswering,
        AutoModelForSequenceClassification,
        AutoModelWithLMHead,
    )

    try:
        from transformers import AutoModelForMaskedLM
    except ImportError:
        from transformers import AutoModelWithLMHead as AutoModelForMaskedLM

    # Available since Transformers 4.12.0
    try:
        from transformers import AutoModelForAudioClassification
    except ImportError:

        class AutoModelForAudioClassification:
            pass


if is_tf_available():
    from transformers import (
        TFAutoModel,
        TFAutoModelForQuestionAnswering,
        TFAutoModelForSequenceClassification,
        TFAutoModelWithLMHead,
    )

    try:
        from transformers import TFAutoModelForMaskedLM
    except ImportError:
        from transformers import TFAutoModelWithLMHead as TFAutoModelForMaskedLM


class _BaseOVAutoModelClass(OVPreTrainedModel):
    def __init__(self):
        raise EnvironmentError(
            f"{self.__class__.__name__} is designed to be instantiated "
            f"using the `{self.__class__.__name__}.from_pretrained(pretrained_model_name_or_path)` method."
        )


class OVAutoModel(_BaseOVAutoModelClass):
    if is_torch_available():
        _pt_auto_model = AutoModel
    if is_tf_available():
        _tf_auto_model = TFAutoModel


class OVAutoModelForMaskedLM(_BaseOVAutoModelClass):
    if is_torch_available():
        _pt_auto_model = AutoModelForMaskedLM
    if is_tf_available():
        _tf_auto_model = TFAutoModelForMaskedLM


class OVAutoModelWithLMHead(_BaseOVAutoModelClass):
    if is_torch_available():
        _pt_auto_model = AutoModelWithLMHead
    if is_tf_available():
        _tf_auto_model = TFAutoModelWithLMHead


class OVAutoModelForQuestionAnswering(_BaseOVAutoModelClass):
    if is_torch_available():
        _pt_auto_model = AutoModelForQuestionAnswering
    if is_tf_available():
        _tf_auto_model = TFAutoModelForQuestionAnswering


class OVAutoModelForSequenceClassification(_BaseOVAutoModelClass):
    if is_torch_available():
        _pt_auto_model = AutoModelForSequenceClassification
    if is_tf_available():
        _tf_auto_model = TFAutoModelForSequenceClassification


class OVAutoModelForAudioClassification(_BaseOVAutoModelClass):
    if is_torch_available():
        _pt_auto_model = AutoModelForAudioClassification

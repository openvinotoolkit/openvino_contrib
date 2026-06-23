# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .base_wrapper import BaseWrapper
from .classification_wrapper import ClassificationWrapper
from .regression_wrapper import RegressionWrapper
from .segmentation_wrapper import SegmentationWrapper
from .detection_wrapper import DetectionWrapper

__all__ = [
    "BaseWrapper",
    "ClassificationWrapper",
    "RegressionWrapper",
    "SegmentationWrapper",
    "DetectionWrapper",
]
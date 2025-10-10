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
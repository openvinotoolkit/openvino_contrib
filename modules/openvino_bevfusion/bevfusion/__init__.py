# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
BEVFusion — Camera + LiDAR 3D object detection.

Usage::

    from bevfusion.pipeline import BEVFusionPipeline
    pipeline = BEVFusionPipeline(model_dir='openvino_model')
    boxes, timings = pipeline.run(sample_info)
"""


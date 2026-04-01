"""
BEVFusion — Camera + LiDAR 3D object detection.

Usage::

    from bevfusion.pipeline import BEVFusionPipeline
    pipeline = BEVFusionPipeline(model_dir='openvino_model')
    boxes, timings = pipeline.run(sample_info)
"""


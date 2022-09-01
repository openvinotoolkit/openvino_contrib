#!/usr/bin/env python3


cfg_overridden_precisions = {
    # Convolution: output value inaccuracy exceeds threshold for FP16
    'vgg16-IR:opid42': ['FP32'],
    'vgg16-IR:opid31': ['FP32'],
    'yolo-v4-tf:opid227': ['FP32'],

    # FusedConvolution: output value inaccuracy exceeds threshold for FP16
    '3d_unet-graph-transform-cuda:opid31': ['FP32'],

}

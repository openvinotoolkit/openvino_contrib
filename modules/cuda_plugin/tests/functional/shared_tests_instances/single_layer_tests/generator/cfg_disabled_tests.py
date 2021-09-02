#!/usr/bin/env python3


cfg_disabled_tests = {
    #
    # Convolutions with Asymmetric padding are not supported:
    'efficientdet-d1-tf:opid6',
    'mask_rcnn_inception_v2_coco:opid104',
    'mask_rcnn_inception_v2_coco:opid119',
    'mask_rcnn_inception_v2_coco:opid578',
    'ssd_mobilenet_v2_coco:opid6',
    'yolo-v3-tf:opid8',
    'yolo-v3-tf:opid27',
    'yolo-v3-tf:opid59',
    'yolo-v3-tf:opid169',
    'yolo-v3-tf:opid279',
    'yolo-v4-tf:opid7',
    'yolo-v4-tf:opid44',
    'yolo-v4-tf:opid92',
    'yolo-v4-tf:opid212',
    'yolo-v4-tf:opid332',
    'yolo-v4-tf:opid540',
    'yolo-v4-tf:opid588',

    #
    # MatMul: terminate called without an active exception
    'LPCnet-lpcnet_enc:opid24',

    #
    # MaxPool: The end corner is out of bounds at axis 3
    'squeezenet1.1:opid41',
    'squeezenet1.1:opid74',
}

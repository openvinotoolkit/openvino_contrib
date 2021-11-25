import torch

from ..hooks import OpenVINOTensor, forward_hook

def rpn_forward(self, images, features, targets=None):
    from torchvision.models.detection.rpn import concat_box_prediction_layers
    # Copy of torchvision/models/detection/rpn.py
    features = list(features.values())
    objectness, pred_bbox_deltas = self.head(features)
    anchors = self.anchor_generator(images, features)

    objectness, pred_bbox_deltas = concat_box_prediction_layers(objectness, pred_bbox_deltas)
    # end of copy

    # Create an alias
    class DetectionOutput(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.variance_encoded_in_target = True
            self.nms_threshold = 0.7
            self.confidence_threshold = -999
            self.top_k = 6000
            self.keep_top_k = 1000
            self.code_type = 'caffe.PriorBoxParameter.CENTER_SIZE'

    # outputs = [OpenVINOTensor(), OpenVINOTensor()]
    # for out in outputs:
    #     out.graph = objectness.graph

    # torch.Size([1, 1, 76824])
    # (1, 76824)
    # (1, 1536480)
    # print(anchors.shape)
    # print(deltas.shape)
    # print(logist.shape)
    anchors = anchors[0].reshape(1, 1, -1)
    anchors /= 320
    pred_bbox_deltas = pred_bbox_deltas.reshape(1, -1)
    objectness = objectness.reshape(1, -1)

    proposals = forward_hook(DetectionOutput(), (pred_bbox_deltas, objectness, OpenVINOTensor(anchors)))
    proposals = proposals.reshape(-1, 7)[:, 3:]
    return [proposals, OpenVINOTensor()]

    # # Return unprocessed deltas, scores and anchors
    # return anchors, pred_bbox_deltas, objectness


class MaskRCNN(object):
    def __init__(self):
        self.class_name = 'torchvision.models.detection.mask_rcnn.MaskRCNN'


    def register_hook(self, model):
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        model.rpn.forward = lambda *args: rpn_forward(model.rpn, *args)
        # model.forward = self.hook(forward, model, model.forward)
        # model.preprocess_image = self.hook(preprocess_image, model, model.preprocess_image)
        # import detectron2
        # detectron2.modeling.meta_arch.retinanet.detector_postprocess = detector_postprocess

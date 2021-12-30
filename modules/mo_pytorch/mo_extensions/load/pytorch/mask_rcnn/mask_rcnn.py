import numpy as np
import torch

from ..hooks import OpenVINOTensor, forward_hook

class DetectionOutput(torch.nn.Module):
    def __init__(self, top_k, nms_threshold, confidence_threshold, background_label_id):
        super().__init__()
        self.variance_encoded_in_target = True
        self.nms_threshold = nms_threshold
        self.confidence_threshold = confidence_threshold
        self.top_k = top_k
        self.keep_top_k = top_k
        self.code_type = 'caffe.PriorBoxParameter.CENTER_SIZE'
        self.background_label_id = background_label_id
        self.clip_before_nms = True
        self.share_location = False

    def infer_shapes(self, inputs):
        return [1, 1, self.keep_top_k, 7]


def filter_detections(detections, is_dynamic):
    if is_dynamic:
        scores = detections[0, 0, :, 2]
        ids = torch.nonzero(scores).reshape(-1)
        return torch.gather(detections, 2, ids)
    else:
        return detections

def rpn_forward(self, is_dynamic, images, features, targets=None):
    from torchvision.models.detection.rpn import concat_box_prediction_layers
    # Copy of torchvision/models/detection/rpn.py
    features = list(features.values())
    objectness, pred_bbox_deltas = self.head(features)
    anchors = self.anchor_generator(images, features)

    shapes = [np.prod(o.shape) for o in objectness]

    objectness, pred_bbox_deltas = concat_box_prediction_layers(objectness, pred_bbox_deltas)
    # end of copy

    img_h, img_w = images.tensors.shape[2:]
    anchors = anchors[0].reshape(1, -1, 4)
    anchors /= torch.tensor([img_w, img_h, img_w, img_h])
    pred_bbox_deltas = pred_bbox_deltas.reshape(1, -1, 4)
    objectness = objectness.reshape(1, -1).sigmoid()

    start_idx = 0
    all_proposals = []
    for shape in shapes:
        end_idx = start_idx + shape
        scores = objectness[:, start_idx : end_idx]
        deltas = pred_bbox_deltas[:, start_idx : end_idx].reshape(1, -1)
        priors = anchors[:, start_idx : end_idx].reshape(1, 1, -1)

        det = DetectionOutput(top_k=min(shape, self.post_nms_top_n()),
                              nms_threshold=self.nms_thresh,
                              confidence_threshold=0.0,
                              background_label_id=2)
        proposals = forward_hook(det, (deltas, scores, OpenVINOTensor(priors)))
        proposals = filter_detections(proposals, is_dynamic)

        all_proposals.append(proposals)
        start_idx = end_idx

    all_proposals = torch.cat(all_proposals, dim=2)

    _, ids = torch.topk(all_proposals[0, 0, :, 2], self.post_nms_top_n())
    all_proposals = torch.gather(all_proposals, 2, ids).reshape(-1, 7)[:, 3:]
    return [all_proposals, OpenVINOTensor()]


def multi_scale_roi_align(cls, features, proposals, image_shapes, is_dynamic):
    num_proposals = proposals.shape[0]
    pooled_h = cls.output_size[0]
    pooled_w = cls.output_size[1]
    ch = features[cls.featmap_names[0]].shape[1]

    # Proposals are in absolute coordinates
    img_h, img_w = image_shapes[0]
    proposals = proposals * torch.tensor([img_w, img_h, img_w, img_h])

    if cls.scales is None:
        x_filtered = [features[k] for k in cls.featmap_names]
        cls.setup_scales(x_filtered, image_shapes)

    levels = cls.map_levels([proposals])

    if is_dynamic:
        # To prevent dynamic layer with zero dimension, add fake indices and proposals
        # so there would be at least one proposal for each level
        num_fake_proposals = len(cls.featmap_names)
        levels = torch.cat((levels, torch.arange(0, num_fake_proposals)))
        proposals = torch.nn.functional.pad(proposals, (0, 0, 0, num_fake_proposals))
        num_proposals += num_fake_proposals
    else:
        levels = levels.reshape(-1, 1, 1, 1)
        zeros = OpenVINOTensor(torch.zeros([num_proposals], dtype=torch.int32))

    final_box_features = OpenVINOTensor(torch.zeros([num_proposals, ch, pooled_h, pooled_w], dtype=torch.float32))
    for lvl, name in enumerate(cls.featmap_names):
        class ROIAlign(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.pooled_h = pooled_h
                self.pooled_w = pooled_w
                self.sampling_ratio = 2
                self.mode = 'avg'
                self.spatial_scale = cls.scales[lvl]

            def infer_shapes(self, inputs):
                feat_shape = inputs[0].dynamic_shape
                rois_shape = inputs[1].dynamic_shape
                return [rois_shape[0], feat_shape[1], self.pooled_h, self.pooled_w]

        if is_dynamic:
            ids = torch.nonzero(levels == lvl).reshape(-1)
            zeros = ids * 0
            proposals_lvl = torch.gather(proposals, 0, ids)
            box_features = forward_hook(ROIAlign(), (features[name], proposals_lvl, zeros))

            ids = ids.reshape(-1, 1, 1, 1).repeat(1, ch, pooled_h, pooled_w)
            final_box_features = torch.scatter(final_box_features, 0, ids, box_features)
        else:
            box_features = forward_hook(ROIAlign(), (features[name], proposals, zeros))

            # Imitation of level-wise ROIAlign
            box_features = box_features * (levels == lvl).to(torch.float32)
            if lvl > 0:
                final_box_features += box_features
            else:
                final_box_features = box_features

    if is_dynamic:
        final_box_features = final_box_features[:-num_fake_proposals]
    return final_box_features


def roi_heads_forward(self, is_dynamic, features, proposals, image_shapes, targets=None):
    box_features = multi_scale_roi_align(self.box_roi_pool, features, proposals, image_shapes, is_dynamic)
    box_features = self.box_head(box_features)
    class_logits, box_regression = self.box_predictor(box_features)

    proposal = proposals.reshape(1, 1, -1)
    box_regression = box_regression.reshape(1, -1, 4)
    box_regression /= self.box_coder.weights
    box_regression = box_regression.reshape(1, -1)
    class_logits = class_logits.softmax(1).reshape(1, -1)

    det = DetectionOutput(top_k=self.detections_per_img,
                          nms_threshold=self.nms_thresh,
                          confidence_threshold=self.score_thresh,
                          background_label_id=0)
    detections = forward_hook(det, (box_regression, class_logits, proposal))

    # Predict masks
    detections = filter_detections(detections, is_dynamic)

    boxes = detections[0, 0, :, 3:]
    mask_features = multi_scale_roi_align(self.mask_roi_pool, features, boxes, image_shapes, is_dynamic)
    mask_features = self.mask_head(mask_features)
    mask_logits = self.mask_predictor(mask_features)
    mask_probs = mask_logits.sigmoid()

    return {'boxes': detections.clone(), 'masks': mask_probs}, {}


def model_forward(self, images, targets=None):
    from torchvision.models.detection.image_list import ImageList
    images = self.transform.normalize(images)
    img_h, img_w = images.shape[2:]
    original_image_sizes = [(img_h, img_w)]
    images = ImageList(images, [[img_h, img_w]])

    features = self.backbone(images.tensors)
    proposals, proposal_losses = self.rpn(images, features, targets)
    detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
    return detections


class MaskRCNN(object):
    def __init__(self):
        self.class_name = 'torchvision.models.detection.mask_rcnn.MaskRCNN'


    def register_hook(self, model, is_dynamic):
        model.forward = lambda *args: model_forward(model, *args)
        model.rpn.forward = lambda *args: rpn_forward(model.rpn, is_dynamic, *args)
        model.roi_heads.forward = lambda *args: roi_heads_forward(model.roi_heads, is_dynamic, *args)

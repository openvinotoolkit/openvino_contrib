import argparse

import numpy as np
import cv2 as cv
import torchvision.models as models
import mo_pytorch
from openvino.inference_engine import IECore

np.random.seed(324)

parser = argparse.ArgumentParser("Sample which runs Mask R-CNN from Torchvision using OpenVINO")
parser.add_argument("-i", dest="input", type=str, required=True,
                    help="Input image to process")
parser.add_argument("-o", dest="output", type=str, default="out.png",
                    help="Name of output image file")
parser.add_argument("-d", dest="device", type=str, default="CPU",
                    help="Target device")
parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                    help="A threshold for detection scores")
parser.add_argument("-mt", "--mask_threshold", type=float, default=0.5,
                    help="A threshold for masks probabilities")
args = parser.parse_args()

# source: https://pytorch.org/vision/master/auto_examples/plot_visualization_utils.html#instance-seg-output
inst_classes = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Load origin model
model = models.detection.mask_rcnn.maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Convert model to OpenVINO IR
img_size = 800
mo_pytorch.convert(model,
                   input_shape=[1, 3, img_size, img_size],
                   model_name="Mask_RCNN",
                   scale=255)

# Prepare input image
img = cv.imread(args.input)
inp = cv.resize(img, (img_size, img_size))
inp = np.expand_dims(inp.astype(np.float32).transpose(2, 0, 1), axis=0)

# Do inference
ie = IECore()
net = ie.read_network('Mask_RCNN.xml')
exec_net = ie.load_network(net, args.device)
out = exec_net.infer({'input': inp})
detections, masks, _ = out.values()

# Network produces two outputs:
# 1) detections with shape 1x1xNx7
# 2) masks with shape NxCxHxW
# <N> is a number of detections,
# <C> is a number of classes
# <H> and <W> are spatil resolution of predicted masks.
# Every mask should be upsampled to bounding box.
# Every detection is a vector of [batch_id, class_id, score, left, top, right, bottom].
color = np.zeros([3], np.uint8)
for detection, mask in zip(detections.reshape(-1, 7), masks):
    class_id, confidence, xmin, ymin, xmax, ymax = detection[1:]
    if confidence < args.prob_threshold:
        continue

    class_id = int(class_id)
    color = (color + np.random.randint(127, 256, [3], np.uint8)) // 2

    # Normalize bounding box to image size
    xmin = int(xmin * (img.shape[1] - 1))
    ymin = int(ymin * (img.shape[0] - 1))
    xmax = int(xmax * (img.shape[1] - 1))
    ymax = int(ymax * (img.shape[0] - 1))

    # Draw a bounding box
    cv.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0))

    # Draw a class name
    label = inst_classes[class_id]
    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(ymin, labelSize[1])
    cv.rectangle(img, (xmin, top - labelSize[1]), (xmin + labelSize[0], top + baseLine), (255, 255, 255), cv.FILLED)
    cv.putText(img, label, (xmin, top), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    # Upsample mask of the predicted class
    mask = mask[class_id]
    mask = cv.resize(mask, (xmax - xmin + 1, ymax - ymin + 1))
    mask = mask > args.mask_threshold

    # mask_color = np.array([255, 0, 0])
    roi = img[ymin : ymax + 1, xmin : xmax + 1][mask]
    img[ymin : ymax + 1, xmin : xmax + 1][mask] = (0.6 * color + 0.4 * roi).astype(np.uint8)

cv.imwrite(args.output, img)

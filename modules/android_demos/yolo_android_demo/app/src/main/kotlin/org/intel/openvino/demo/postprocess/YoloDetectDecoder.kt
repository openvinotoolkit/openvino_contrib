// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package org.intel.openvino.demo.postprocess

import org.intel.openvino.demo.inference.ModelConfig
import org.intel.openvino.demo.preprocess.LetterboxTransform

/**
 * Anchor-free YOLOv8/YOLO11 detection decoder.
 *
 * Single output `[1, 4 + numClasses, numBoxes]` (e.g. `[1, 84, 8400]` for COCO): boxes as
 * `cx, cy, w, h` in model-input pixel space, per-class scores with no objectness. Confidence
 * filter → class-aware NMS → inverse letterbox to frame coordinates.
 */
class YoloDetectDecoder : YoloDecoder {

    override fun decode(
        outputs: List<RawOutput>,
        transform: LetterboxTransform,
        config: ModelConfig,
        labels: List<String>,
        confThreshold: Float,
        iouThreshold: Float,
    ): InferenceResult {
        val out = outputs[0]
        // shape = [1, channels, numBoxes]
        val channels = out.shape[1]
        val numBoxes = out.shape[2]
        val numClasses = channels - 4

        val candidates = DetectHead.collect(out.data, channels, numBoxes, numClasses, confThreshold)

        val nmsInput = candidates.mapIndexed { i, c ->
            val box = DetectHead.toFrameBox(c, transform)
            NmsBox(box.left, box.top, box.right, box.bottom, c.score, c.classId, i)
        }
        val kept = Nms.classAware(nmsInput, iouThreshold)

        val detections = kept.map { nms ->
            Detection(
                box = Box(nms.left, nms.top, nms.right, nms.bottom),
                classId = nms.classId,
                score = nms.score,
                label = DetectHead.labelFor(labels, nms.classId),
            )
        }
        return InferenceResult(detections = detections)
    }
}

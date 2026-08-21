// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package org.intel.openvino.demo.postprocess

import org.intel.openvino.demo.inference.ModelConfig
import org.intel.openvino.demo.preprocess.LetterboxTransform

/**
 * Anchor-free YOLOv8/YOLO11 pose decoder.
 *
 * Single output `[1, 4 + 1 + K*3, numBoxes]` (e.g. `[1, 56, 8400]` = 4 box + 1 person class +
 * 17 keypoints x (x, y, visibility)). Boxes decode like detection; keypoints are read from the
 * trailing channels and mapped back through the inverse letterbox.
 */
class YoloPoseDecoder : YoloDecoder {

    override fun decode(
        outputs: List<RawOutput>,
        transform: LetterboxTransform,
        config: ModelConfig,
        labels: List<String>,
        confThreshold: Float,
        iouThreshold: Float,
    ): InferenceResult {
        val out = outputs[0]
        val channels = out.shape[1]
        val numBoxes = out.shape[2]
        val numClasses = config.numClasses // pose models have a single 'person' class
        // Channels after box(4) + classes are keypoints, 3 values each.
        val kptChannels = channels - 4 - numClasses
        val numKeypoints = kptChannels / 3

        val candidates = DetectHead.collect(out.data, channels, numBoxes, numClasses, confThreshold)

        val nmsInput = candidates.mapIndexed { i, c ->
            val box = DetectHead.toFrameBox(c, transform)
            NmsBox(box.left, box.top, box.right, box.bottom, c.score, c.classId, i)
        }
        val kept = Nms.classAware(nmsInput, iouThreshold)

        val kptBase = (4 + numClasses) * numBoxes
        val poses = kept.map { nms ->
            val c = candidates[nms.sourceIndex]
            val anchor = c.anchorIndex
            val points = ArrayList<Keypoint>(numKeypoints)
            for (k in 0 until numKeypoints) {
                val xIdx = kptBase + (k * 3 + 0) * numBoxes + anchor
                val yIdx = kptBase + (k * 3 + 1) * numBoxes + anchor
                val vIdx = kptBase + (k * 3 + 2) * numBoxes + anchor
                val fx = transform.frameX(out.data[xIdx])
                val fy = transform.frameY(out.data[yIdx])
                points.add(Keypoint(fx, fy, out.data[vIdx]))
            }
            Keypoints(
                detection = Detection(
                    box = Box(nms.left, nms.top, nms.right, nms.bottom),
                    classId = nms.classId,
                    score = nms.score,
                    label = DetectHead.labelFor(labels, nms.classId),
                ),
                points = points,
            )
        }
        return InferenceResult(poses = poses)
    }
}

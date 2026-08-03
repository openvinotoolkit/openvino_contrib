// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package org.intel.openvino.demo.postprocess

import org.intel.openvino.demo.inference.ModelConfig
import org.intel.openvino.demo.preprocess.LetterboxTransform

/**
 * YOLOv10 end-to-end (NMS-free) detection decoder.
 *
 * Unlike the anchor-free v8/v11 head (`[1, 4+nc, N]` of raw candidates needing NMS), YOLOv10 emits
 * a single already-finalized output `[1, 300, 6]`: up to 300 rows of `[x1, y1, x2, y2, score,
 * classId]`, corner coordinates in model-input pixel space, sorted by descending score, with the
 * dual-label-assignment NMS folded into the graph. So there is **no NMS here** — just a confidence
 * threshold and the inverse letterbox. This keeps the anchor-free single-path decoder untouched;
 * v10's different layout lives entirely in this class (selected via the model's decoder family).
 */
class YoloV10DetectDecoder : YoloDecoder {

    override fun decode(
        outputs: List<RawOutput>,
        transform: LetterboxTransform,
        config: ModelConfig,
        labels: List<String>,
        confThreshold: Float,
        iouThreshold: Float,
    ): InferenceResult {
        val out = outputs[0]
        // shape = [1, numDet, 6]; flat row stride is 6.
        val numDet = out.shape[1]
        val stride = out.shape[2] // 6
        val data = out.data

        val fw = transform.frameWidth.toFloat()
        val fh = transform.frameHeight.toFloat()

        val detections = ArrayList<Detection>()
        var i = 0
        while (i < numDet) {
            val base = i * stride
            val score = data[base + 4]
            // Rows are score-descending, so we can stop at the first sub-threshold row.
            if (score < confThreshold) break

            val classId = data[base + 5].toInt()
            // Corner coords in model-input pixels -> frame pixels via the inverse letterbox.
            val left = transform.frameX(data[base + 0]).coerceIn(0f, fw)
            val top = transform.frameY(data[base + 1]).coerceIn(0f, fh)
            val right = transform.frameX(data[base + 2]).coerceIn(0f, fw)
            val bottom = transform.frameY(data[base + 3]).coerceIn(0f, fh)

            detections.add(
                Detection(
                    box = Box(left, top, right, bottom),
                    classId = classId,
                    score = score,
                    label = DetectHead.labelFor(labels, classId),
                )
            )
            i++
        }
        return InferenceResult(detections = detections)
    }
}

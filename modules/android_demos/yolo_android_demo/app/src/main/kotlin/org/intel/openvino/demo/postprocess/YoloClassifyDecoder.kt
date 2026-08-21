// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package org.intel.openvino.demo.postprocess

import org.intel.openvino.demo.inference.ModelConfig
import org.intel.openvino.demo.preprocess.LetterboxTransform
import kotlin.math.exp

/**
 * YOLOv8/YOLO11 classification decoder.
 *
 * Single output `[1, numClasses]` (e.g. `[1, 1000]` for ImageNet). Softmax over the logits, then
 * the top-[topK] classes. No boxes are produced.
 */
class YoloClassifyDecoder(private val topK: Int = 5) : YoloDecoder {

    override fun decode(
        outputs: List<RawOutput>,
        transform: LetterboxTransform,
        config: ModelConfig,
        labels: List<String>,
        confThreshold: Float,
        iouThreshold: Float,
    ): InferenceResult {
        val logits = outputs[0].data

        // Numerically stable softmax.
        var max = Float.NEGATIVE_INFINITY
        for (v in logits) if (v > max) max = v
        var sum = 0.0
        val probs = FloatArray(logits.size)
        for (i in logits.indices) {
            val e = exp((logits[i] - max).toDouble())
            probs[i] = e.toFloat()
            sum += e
        }
        val inv = if (sum > 0.0) (1.0 / sum).toFloat() else 0f

        // Top-k by probability.
        val indices = logits.indices.sortedByDescending { probs[it] }.take(topK)
        val classifications = indices.map { idx ->
            Classification(
                classId = idx,
                score = probs[idx] * inv,
                label = DetectHead.labelFor(labels, idx),
            )
        }
        return InferenceResult(classifications = classifications)
    }
}

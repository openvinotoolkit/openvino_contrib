// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package org.intel.openvino.demo.postprocess

import org.intel.openvino.demo.inference.ModelConfig
import org.intel.openvino.demo.preprocess.LetterboxTransform

/**
 * A single output tensor read back from inference: its flat float data and its shape. Fetched by
 * index (Ultralytics OpenVINO exports leave outputs unnamed), matching the order in
 * [ModelConfig.outputs].
 */
data class RawOutput(val data: FloatArray, val shape: IntArray) {
    override fun equals(other: Any?): Boolean =
        this === other || (other is RawOutput && data.contentEquals(other.data) && shape.contentEquals(other.shape))

    override fun hashCode(): Int = 31 * data.contentHashCode() + shape.contentHashCode()
}

/**
 * Decodes raw model outputs into [InferenceResult]. All YOLO version/task specifics live behind
 * this interface; the camera/preprocess/inference/overlay layers are decoder-agnostic. Adding a
 * new YOLO variant means adding a config + (maybe) a decoder, nothing else.
 *
 * Implementations support **anchor-free YOLOv8/YOLO11 only** — per-class scores with no separate
 * objectness column. There is deliberately no legacy anchor-based (objectness) code path.
 */
interface YoloDecoder {
    /**
     * @param outputs raw output tensors, in [ModelConfig.outputs] index order
     * @param transform the letterbox transform used for this frame (to invert to frame coordinates)
     * @param config the model configuration
     * @param labels class labels
     * @param confThreshold minimum score to keep a detection
     * @param iouThreshold IoU threshold for NMS
     */
    fun decode(
        outputs: List<RawOutput>,
        transform: LetterboxTransform,
        config: ModelConfig,
        labels: List<String>,
        confThreshold: Float,
        iouThreshold: Float,
    ): InferenceResult

    companion object {
        /** Output-format families a model can belong to (from the manifest `variant`). */
        const val FAMILY_ANCHOR_FREE = "anchor-free"   // YOLOv8 / YOLO11: [1, 4+nc, N]
        const val FAMILY_YOLOV10_E2E = "yolov10-e2e"    // YOLOv10: end-to-end NMS-free [1, 300, 6]

        /**
         * Pick the decoder from the model's output-format family + task. Adding a YOLO variant with
         * a new output layout is a new family + decoder here; existing families/decoders are reused.
         */
        fun forConfig(config: org.intel.openvino.demo.inference.ModelConfig): YoloDecoder {
            val task = config.task
            return when (config.decoder) {
                FAMILY_YOLOV10_E2E -> YoloV10DetectDecoder() // v10 publishes detect only
                else -> when (task) { // anchor-free (v8/v11) default
                    org.intel.openvino.demo.inference.YoloTask.DETECT -> YoloDetectDecoder()
                    org.intel.openvino.demo.inference.YoloTask.SEGMENT -> YoloSegmentDecoder()
                    org.intel.openvino.demo.inference.YoloTask.POSE -> YoloPoseDecoder()
                    org.intel.openvino.demo.inference.YoloTask.CLASSIFY -> YoloClassifyDecoder()
                }
            }
        }
    }
}

/**
 * Shared helpers for the channels-first anchor-free detection head, used by detect / segment / pose.
 *
 * The detection output has layout `[1, C, N]` (channels-first, e.g. `[1, 84, 8400]`): for anchor
 * box `n`, channel `c` is read at flat index `c * N + n`. The first 4 channels are the box as
 * `cx, cy, w, h` in **model-input pixel space**; the next `numClasses` channels are per-class
 * scores (already probabilities, no objectness). Any channels beyond that (mask coefficients,
 * keypoints) are task-specific and handled by the concrete decoder.
 */
internal object DetectHead {

    /** A raw candidate carrying its winning class, score, box (model space), and anchor index. */
    data class Candidate(
        val cx: Float,
        val cy: Float,
        val w: Float,
        val h: Float,
        val classId: Int,
        val score: Float,
        val anchorIndex: Int,
    )

    /**
     * Scan the channels-first detection tensor and return confidence-passing candidates.
     *
     * @param data flat output data, length `channels * numBoxes`
     * @param channels total channels C (e.g. 84 for detect, 116 for seg, 56 for pose)
     * @param numBoxes number of anchors N (e.g. 8400)
     * @param numClasses number of class channels (channels 4 .. 4+numClasses)
     */
    fun collect(
        data: FloatArray,
        channels: Int,
        numBoxes: Int,
        numClasses: Int,
        confThreshold: Float,
    ): List<Candidate> {
        // Guard against a manifest/shape mismatch: the flat buffer must hold C*N values,
        // and the class channels (4 .. 4+numClasses) must fit within C.
        require(data.size >= channels * numBoxes) {
            "output too small: ${data.size} < channels*numBoxes ($channels*$numBoxes)"
        }
        require(4 + numClasses <= channels) {
            "class channels (4+$numClasses) exceed channel count $channels"
        }

        val candidates = ArrayList<Candidate>()
        // Channel base offsets into the flat channels-first buffer.
        val cxBase = 0 * numBoxes
        val cyBase = 1 * numBoxes
        val wBase = 2 * numBoxes
        val hBase = 3 * numBoxes
        val classBase = 4 * numBoxes

        for (n in 0 until numBoxes) {
            // Find the best class for this anchor.
            var bestClass = -1
            var bestScore = confThreshold
            var c = 0
            while (c < numClasses) {
                val score = data[classBase + c * numBoxes + n]
                if (score > bestScore) {
                    bestScore = score
                    bestClass = c
                }
                c++
            }
            if (bestClass < 0) continue

            candidates.add(
                Candidate(
                    cx = data[cxBase + n],
                    cy = data[cyBase + n],
                    w = data[wBase + n],
                    h = data[hBase + n],
                    classId = bestClass,
                    score = bestScore,
                    anchorIndex = n,
                )
            )
        }
        return candidates
    }

    /** Convert a model-space center box to a frame-space [Box] via the inverse letterbox. */
    fun toFrameBox(candidate: Candidate, transform: LetterboxTransform): Box {
        val modelLeft = candidate.cx - candidate.w / 2f
        val modelTop = candidate.cy - candidate.h / 2f
        val modelRight = candidate.cx + candidate.w / 2f
        val modelBottom = candidate.cy + candidate.h / 2f
        return Box(
            left = transform.frameX(modelLeft).coerceIn(0f, transform.frameWidth.toFloat()),
            top = transform.frameY(modelTop).coerceIn(0f, transform.frameHeight.toFloat()),
            right = transform.frameX(modelRight).coerceIn(0f, transform.frameWidth.toFloat()),
            bottom = transform.frameY(modelBottom).coerceIn(0f, transform.frameHeight.toFloat()),
        )
    }

    fun labelFor(labels: List<String>, classId: Int): String =
        labels.getOrElse(classId) { "class $classId" }
}

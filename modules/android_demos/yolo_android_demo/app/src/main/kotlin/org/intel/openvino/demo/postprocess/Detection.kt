// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package org.intel.openvino.demo.postprocess

/**
 * Result types produced by the YOLO decoders.
 *
 * All coordinates in the public result types are in **original frame pixel space** (the camera
 * frame as displayed), already mapped back through the inverse letterbox transform. The camera,
 * preprocess, inference and overlay layers never see YOLO-specific data; they only see these types.
 */

/** An axis-aligned bounding box in frame-pixel coordinates. */
data class Box(
    val left: Float,
    val top: Float,
    val right: Float,
    val bottom: Float,
) {
    val width: Float get() = right - left
    val height: Float get() = bottom - top
}

/** A single detected object. */
data class Detection(
    val box: Box,
    val classId: Int,
    val score: Float,
    val label: String,
)

/**
 * A per-object instance mask, stored as a small single-channel probability grid in frame-pixel
 * space along with the box it belongs to. The overlay upsamples it when drawing.
 */
data class Mask(
    val detection: Detection,
    /** Row-major mask values in [0,1], size [maskHeight * maskWidth]. */
    val data: FloatArray,
    val maskWidth: Int,
    val maskHeight: Int,
    /** The region of the frame the mask grid covers (usually the full frame). */
    val region: Box,
) {
    override fun equals(other: Any?): Boolean =
        this === other ||
            (other is Mask &&
                detection == other.detection &&
                maskWidth == other.maskWidth &&
                maskHeight == other.maskHeight &&
                region == other.region &&
                data.contentEquals(other.data))

    override fun hashCode(): Int {
        var result = detection.hashCode()
        result = 31 * result + maskWidth
        result = 31 * result + maskHeight
        result = 31 * result + region.hashCode()
        result = 31 * result + data.contentHashCode()
        return result
    }
}

/** A single keypoint with a visibility/confidence value. */
data class Keypoint(val x: Float, val y: Float, val score: Float)

/** A detected person plus its skeleton keypoints. */
data class Keypoints(
    val detection: Detection,
    val points: List<Keypoint>,
)

/** A single top-k classification entry. */
data class Classification(
    val classId: Int,
    val score: Float,
    val label: String,
)

/**
 * The decoded output of one inference, tagged by task. Exactly one list is populated depending on
 * the model's task; the overlay renders whichever is present.
 */
data class InferenceResult(
    val detections: List<Detection> = emptyList(),
    val masks: List<Mask> = emptyList(),
    val poses: List<Keypoints> = emptyList(),
    val classifications: List<Classification> = emptyList(),
)

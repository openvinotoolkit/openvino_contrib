// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package org.intel.openvino.demo.inference

import org.intel.openvino.demo.data.ModelEntry

/** The YOLO task family a model performs. */
enum class YoloTask {
    DETECT,
    SEGMENT,
    POSE,
    CLASSIFY;

    companion object {
        fun fromString(value: String): YoloTask =
            when (value.lowercase()) {
                "detect" -> DETECT
                "segment" -> SEGMENT
                "pose" -> POSE
                "classify" -> CLASSIFY
                else -> throw IllegalArgumentException("Unknown YOLO task: $value")
            }
    }
}

/**
 * Per-model metadata used by the inference + decode path, built from a manifest [ModelEntry] plus
 * the resolved class [labels]. This is the single place that knows "this model is YOLOv8-seg,
 * 640×640, NCHW, /255, these labels" — nothing YOLO-specific is hardcoded in the
 * camera/preprocess/inference/overlay layers. Adding a model is a manifest entry + (maybe) reusing
 * an existing decoder.
 */
data class ModelConfig(
    val id: String,
    val displayName: String,
    val task: YoloTask,
    /** Output-format family, drives decoder choice: "anchor-free" (v8/v11) or "yolov10-e2e" (v10). */
    val decoder: String,
    val onnxPath: String,
    /** Model input size actually used this session (for dynamic models, derived from the frame). */
    val inputWidth: Int,
    val inputHeight: Int,
    val layout: String,
    /** Per-channel divisor applied inside the compiled model (e.g. 255 for /255). */
    val scale: FloatArray,
    /** Per-channel mean subtracted inside the compiled model (0 for YOLO). */
    val mean: FloatArray,
    val labels: List<String>,
    /** True if the ONNX has a dynamic spatial input; the model is reshaped to inputWidth×Height. */
    val dynamic: Boolean,
) {
    val numClasses: Int get() = labels.size

    override fun equals(other: Any?): Boolean = this === other || (other is ModelConfig && id == other.id)

    override fun hashCode(): Int = id.hashCode()

    companion object {
        /** Round to the nearest positive multiple of 32 (YOLO stride), min 32. */
        fun roundToStride(v: Int, stride: Int = 32): Int =
            (Math.round(v.toFloat() / stride) * stride).coerceAtLeast(stride)

        /**
         * Build a config from a downloaded model [entry] at [onnxPath] with its [labels].
         *
         * For a **static** model the input is the fixed square [ModelEntry.imgsz]. For a **dynamic**
         * model the input is derived from the camera frame's aspect: the longer side is
         * [ModelEntry.imgsz] and the shorter side is scaled to the frame aspect, both rounded to a
         * multiple of 32 — so inference runs on a near-frame-shaped input with little/no padding.
         * [frameWidth]/[frameHeight] are the upright camera frame dims (ignored for static models).
         */
        fun from(
            entry: ModelEntry,
            onnxPath: String,
            labels: List<String>,
            frameWidth: Int = 0,
            frameHeight: Int = 0,
        ): ModelConfig {
            val base = entry.imgsz
            val (w, h) = if (entry.dynamic && frameWidth > 0 && frameHeight > 0) {
                if (frameWidth >= frameHeight) {
                    base to roundToStride((base.toFloat() * frameHeight / frameWidth).toInt())
                } else {
                    roundToStride((base.toFloat() * frameWidth / frameHeight).toInt()) to base
                }
            } else {
                base to base
            }
            return ModelConfig(
                id = entry.id,
                displayName = entry.displayName,
                task = YoloTask.fromString(entry.task),
                decoder = entry.variant,
                onnxPath = onnxPath,
                inputWidth = w,
                inputHeight = h,
                layout = entry.layout,
                scale = floatArrayOf(entry.scale, entry.scale, entry.scale),
                mean = floatArrayOf(0f, 0f, 0f),
                labels = labels,
                dynamic = entry.dynamic,
            )
        }
    }
}

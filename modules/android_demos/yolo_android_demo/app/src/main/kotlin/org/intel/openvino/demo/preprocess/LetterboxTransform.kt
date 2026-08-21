// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package org.intel.openvino.demo.preprocess

/**
 * The affine mapping between the upright camera frame and the square, letterboxed model input.
 *
 * YOLO models take a square input (e.g. 640x640). To avoid distorting the aspect ratio, the frame
 * is resized by a single [scale] to fit inside the square and centered with padding ([padX],
 * [padY]). Getting the **inverse** of this transform exactly right is what keeps boxes aligned —
 * it is the single most common source of "boxes are shifted" bugs, so it lives in one place and is
 * unit-tested.
 *
 *   model_x = frame_x * scale + padX
 *   model_y = frame_y * scale + padY
 *
 *   frame_x = (model_x - padX) / scale
 *   frame_y = (model_y - padY) / scale
 *
 * @property frameWidth  width of the upright camera frame (pixels)
 * @property frameHeight height of the upright camera frame (pixels)
 * @property scale       single resize factor applied to the frame to fit the square
 * @property padX        left padding inside the square (pixels), in model space
 * @property padY        top padding inside the square (pixels), in model space
 */
data class LetterboxTransform(
    val frameWidth: Int,
    val frameHeight: Int,
    val scale: Float,
    val padX: Float,
    val padY: Float,
) {
    /** Map an x coordinate from model input space back to upright frame space. */
    fun frameX(modelX: Float): Float = (modelX - padX) / scale

    /** Map a y coordinate from model input space back to upright frame space. */
    fun frameY(modelY: Float): Float = (modelY - padY) / scale

    companion object {
        /** Compute the letterbox for fitting [frameWidth]x[frameHeight] into [inputWidth]x[inputHeight]. */
        fun compute(
            frameWidth: Int,
            frameHeight: Int,
            inputWidth: Int,
            inputHeight: Int,
        ): LetterboxTransform {
            val scale = minOf(
                inputWidth.toFloat() / frameWidth,
                inputHeight.toFloat() / frameHeight,
            )
            val scaledW = frameWidth * scale
            val scaledH = frameHeight * scale
            val padX = (inputWidth - scaledW) / 2f
            val padY = (inputHeight - scaledH) / 2f
            return LetterboxTransform(frameWidth, frameHeight, scale, padX, padY)
        }
    }
}

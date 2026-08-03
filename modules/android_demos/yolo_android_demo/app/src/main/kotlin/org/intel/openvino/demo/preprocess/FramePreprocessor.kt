// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package org.intel.openvino.demo.preprocess

import android.graphics.Bitmap
import androidx.camera.core.ImageProxy

/**
 * Turns a CameraX [ImageProxy] (YUV_420_888) into a `u8` RGB NHWC buffer at the model's square
 * input size, **without OpenCV**. Everything OpenCV used to do here — color conversion, rotation,
 * resize — is plain Kotlin operating directly on the image plane buffers. The remaining steps
 * (divide by 255, NHWC→NCHW transpose) are pushed into the compiled model via `PrePostProcessor`,
 * so this class only emits raw `u8` pixels.
 *
 * The conversion is done by inverse sampling: for every destination pixel in the square input we
 * map back through the letterbox and the frame rotation to a source pixel in the raw YUV buffer,
 * then convert that single pixel. This fuses YUV→RGB + rotate + letterbox into one pass with no
 * intermediate bitmaps.
 */
class FramePreprocessor(
    private val inputWidth: Int,
    private val inputHeight: Int,
) {
    /** Padding gray used by Ultralytics letterboxing. */
    private val padValue: Byte = 114.toByte()

    /** Reused output buffer: [inputHeight * inputWidth * 3] interleaved RGB. */
    private val output = ByteArray(inputWidth * inputHeight * 3)

    // Reused plane scratch buffers, grown on demand.
    private var yBytes = ByteArray(0)
    private var uBytes = ByteArray(0)
    private var vBytes = ByteArray(0)

    /** The transform produced by the most recent [process] call, for decoding output coordinates. */
    lateinit var lastTransform: LetterboxTransform
        private set

    /**
     * Convert [image] and return the reused RGB buffer. The returned array is valid until the next
     * [process] call (single-threaded inference, so this is safe). [rotationDegrees] is the frame's
     * `imageInfo.rotationDegrees` (0/90/180/270) needed to display it upright.
     */
    /**
     * Letterbox a decoded [bitmap] (already upright RGB) into the reused `u8` RGB NHWC buffer,
     * using the exact same transform as the camera path. Used by the static-image test mode so the
     * whole preprocess→infer→decode→render pipeline can be validated deterministically, without the
     * camera. The returned array is valid until the next call.
     */
    fun processBitmap(bitmap: Bitmap): ByteArray {
        val frameW = bitmap.width
        val frameH = bitmap.height
        val transform = LetterboxTransform.compute(frameW, frameH, inputWidth, inputHeight)
        lastTransform = transform

        // Read all source pixels once (ARGB_8888).
        val srcPixels = IntArray(frameW * frameH)
        bitmap.getPixels(srcPixels, 0, frameW, 0, 0, frameW, frameH)

        val out = output
        out.fill(padValue)
        val scale = transform.scale
        val padX = transform.padX
        val padY = transform.padY

        for (dy in 0 until inputHeight) {
            val fy = ((dy + 0.5f) - padY) / scale - 0.5f
            val fyi = Math.round(fy)
            if (fyi < 0 || fyi >= frameH) continue
            var dstIdx = dy * inputWidth * 3
            val srcRow = fyi * frameW
            for (dx in 0 until inputWidth) {
                val fx = ((dx + 0.5f) - padX) / scale - 0.5f
                val fxi = Math.round(fx)
                if (fxi < 0 || fxi >= frameW) {
                    dstIdx += 3
                    continue
                }
                val argb = srcPixels[srcRow + fxi]
                out[dstIdx] = ((argb ushr 16) and 0xFF).toByte()     // R
                out[dstIdx + 1] = ((argb ushr 8) and 0xFF).toByte()  // G
                out[dstIdx + 2] = (argb and 0xFF).toByte()           // B
                dstIdx += 3
            }
        }
        return out
    }

    fun process(image: ImageProxy, rotationDegrees: Int): ByteArray {
        val rawW = image.width
        val rawH = image.height

        // Upright frame dimensions after applying the rotation.
        val frameW: Int
        val frameH: Int
        if (rotationDegrees == 90 || rotationDegrees == 270) {
            frameW = rawH
            frameH = rawW
        } else {
            frameW = rawW
            frameH = rawH
        }

        val transform = LetterboxTransform.compute(frameW, frameH, inputWidth, inputHeight)
        lastTransform = transform

        val planes = image.planes
        val yPlane = planes[0]
        val uPlane = planes[1]
        val vPlane = planes[2]

        val yRowStride = yPlane.rowStride
        val yPixStride = yPlane.pixelStride
        val uvRowStride = uPlane.rowStride
        val uvPixStride = uPlane.pixelStride

        yBytes = copyBuffer(yPlane.buffer, yBytes)
        uBytes = copyBuffer(uPlane.buffer, uBytes)
        vBytes = copyBuffer(vPlane.buffer, vBytes)

        val out = output
        // Prefill with padding gray; only the letterbox interior is overwritten.
        out.fill(padValue)

        val scale = transform.scale
        val padX = transform.padX
        val padY = transform.padY

        for (dy in 0 until inputHeight) {
            // Inverse letterbox on y: destination row -> upright frame y.
            val fy = ((dy + 0.5f) - padY) / scale - 0.5f
            val fyi = Math.round(fy)
            if (fyi < 0 || fyi >= frameH) continue

            var dstIdx = dy * inputWidth * 3
            for (dx in 0 until inputWidth) {
                val fx = ((dx + 0.5f) - padX) / scale - 0.5f
                val fxi = Math.round(fx)
                if (fxi < 0 || fxi >= frameW) {
                    dstIdx += 3
                    continue
                }

                // Inverse rotation: upright frame (fxi, fyi) -> raw buffer (rx, ry).
                val rx: Int
                val ry: Int
                when (rotationDegrees) {
                    90 -> { rx = fyi; ry = rawH - 1 - fxi }
                    180 -> { rx = rawW - 1 - fxi; ry = rawH - 1 - fyi }
                    270 -> { rx = rawW - 1 - fyi; ry = fxi }
                    else -> { rx = fxi; ry = fyi }
                }

                // Sample YUV at (rx, ry).
                val y = yBytes[ry * yRowStride + rx * yPixStride].toInt() and 0xFF
                val uvIndex = (ry shr 1) * uvRowStride + (rx shr 1) * uvPixStride
                val u = (uBytes[uvIndex].toInt() and 0xFF) - 128
                val v = (vBytes[uvIndex].toInt() and 0xFF) - 128

                // YUV (full range) -> RGB.
                val r = y + 1.402f * v
                val g = y - 0.344136f * u - 0.714136f * v
                val b = y + 1.772f * u

                out[dstIdx] = clampToByte(r)
                out[dstIdx + 1] = clampToByte(g)
                out[dstIdx + 2] = clampToByte(b)
                dstIdx += 3
            }
        }
        return out
    }

    private fun copyBuffer(buffer: java.nio.ByteBuffer, scratch: ByteArray): ByteArray {
        buffer.rewind()
        val size = buffer.remaining()
        val dst = if (scratch.size >= size) scratch else ByteArray(size)
        buffer.get(dst, 0, size)
        return dst
    }

    private fun clampToByte(value: Float): Byte {
        val v = value.toInt()
        return when {
            v < 0 -> 0
            v > 255 -> 255.toByte()
            else -> v.toByte()
        }
    }
}

package com.itlab.ai

import android.annotation.SuppressLint
import android.graphics.Bitmap
import androidx.core.graphics.scale

internal class YoloImagePreprocessor(
    private val inputSize: Int,
) {
    @SuppressLint("UseKtx")
    fun preprocess(bitmap: Bitmap): FloatArray {
        val resized = bitmap.scale(inputSize, inputSize)
        val pixels = IntArray(inputSize * inputSize)
        resized.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)
        if (resized !== bitmap) {
            resized.recycle()
        }

        val area = inputSize * inputSize
        val inputData = FloatArray(RGB_CHANNELS * area)
        pixels.forEachIndexed { index, pixel ->
            inputData[index] = (pixel shr RED_SHIFT and CHANNEL_MASK) / CHANNEL_MAX
            inputData[area + index] = (pixel shr GREEN_SHIFT and CHANNEL_MASK) / CHANNEL_MAX
            inputData[2 * area + index] = (pixel and CHANNEL_MASK) / CHANNEL_MAX
        }
        return inputData
    }

    private companion object {
        const val RGB_CHANNELS = 3
        const val RED_SHIFT = 16
        const val GREEN_SHIFT = 8
        const val CHANNEL_MASK = 0xFF
        const val CHANNEL_MAX = 255.0f
    }
}

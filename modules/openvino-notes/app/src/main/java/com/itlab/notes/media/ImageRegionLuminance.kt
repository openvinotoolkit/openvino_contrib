package com.itlab.notes.media

import android.graphics.Bitmap
import android.graphics.drawable.BitmapDrawable
import android.graphics.drawable.Drawable
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.luminance
import androidx.core.graphics.drawable.toBitmap
import androidx.core.graphics.get

/**
 * Estimates whether the top-end region of an image is light (for adaptive icon contrast).
 */
object ImageRegionLuminance {
    private const val LIGHT_REGION_THRESHOLD = 0.55f

    fun isTopEndRegionLight(
        drawable: Drawable,
        sampleMaxSize: Int = 96,
    ): Boolean {
        val width = drawable.intrinsicWidth.takeIf { it > 0 } ?: sampleMaxSize
        val height = drawable.intrinsicHeight.takeIf { it > 0 } ?: sampleMaxSize
        val scale = minOf(1f, sampleMaxSize.toFloat() / maxOf(width, height))
        val targetWidth = (width * scale).toInt().coerceAtLeast(1)
        val targetHeight = (height * scale).toInt().coerceAtLeast(1)
        val bitmap = drawable.toBitmap(targetWidth, targetHeight)
        return try {
            isTopEndRegionLight(bitmap)
        } finally {
            if (drawable !is BitmapDrawable || drawable.bitmap !== bitmap) {
                bitmap.recycle()
            }
        }
    }

    private fun isTopEndRegionLight(bitmap: Bitmap): Boolean {
        val width = bitmap.width
        val height = bitmap.height
        if (width == 0 || height == 0) return true

        val startX = (width * 0.5f).toInt().coerceIn(0, width - 1)
        val endY = (height * 0.5f).toInt().coerceAtLeast(1)
        val step = maxOf(1, minOf(width, height) / 10)

        var luminanceSum = 0.0
        var count = 0
        var x = startX
        while (x < width) {
            var y = 0
            while (y < endY) {
                val pixel = bitmap[x, y]
                val composeColor =
                    Color(
                        red = android.graphics.Color.red(pixel) / 255f,
                        green = android.graphics.Color.green(pixel) / 255f,
                        blue = android.graphics.Color.blue(pixel) / 255f,
                    )
                luminanceSum += composeColor.luminance()
                count++
                y += step
            }
            x += step
        }

        if (count == 0) return true
        return (luminanceSum / count) > LIGHT_REGION_THRESHOLD
    }
}

// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package org.intel.openvino.demo.ui

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import org.intel.openvino.demo.postprocess.InferenceResult
import org.intel.openvino.demo.postprocess.Keypoints
import org.intel.openvino.demo.postprocess.Mask
import kotlin.math.max

/**
 * Draws inference results on top of the CameraX preview using an Android [Canvas] — never via a CV
 * library. Handles boxes, translucent instance masks, pose skeletons + keypoints, top-k
 * classification text, and an FPS/latency readout.
 *
 * Coordinates in [InferenceResult] are in frame-pixel space (frameWidth x frameHeight). The view
 * maps them to view pixels using the same center-crop ("fillCenter") mapping the PreviewView uses,
 * so overlays line up with the live preview.
 */
class OverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
) : View(context, attrs) {

    private var result: InferenceResult = InferenceResult()
    private var frameWidth: Int = 0
    private var frameHeight: Int = 0
    private var fpsText: String = ""

    /** Optional background image, drawn behind overlays in the static-image test mode. */
    private var backgroundBitmap: Bitmap? = null

    private val boxPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 4f
    }
    private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textSize = 36f
        isFakeBoldText = true
    }
    private val textBgPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.argb(160, 0, 0, 0)
        style = Paint.Style.FILL
    }
    private val fpsPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.GREEN
        textSize = 40f
        isFakeBoldText = true
    }
    private val kptPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.YELLOW
        style = Paint.Style.FILL
    }
    private val skeletonPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.CYAN
        strokeWidth = 3f
        style = Paint.Style.STROKE
    }
    private val maskPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        alpha = 120
    }

    // Reused bitmap for mask rendering (grown as needed).
    private var maskBitmap: Bitmap? = null

    /** COCO 17-keypoint skeleton edges (person). */
    private val skeleton = arrayOf(
        intArrayOf(0, 1), intArrayOf(0, 2), intArrayOf(1, 3), intArrayOf(2, 4),
        intArrayOf(5, 6), intArrayOf(5, 7), intArrayOf(7, 9), intArrayOf(6, 8),
        intArrayOf(8, 10), intArrayOf(5, 11), intArrayOf(6, 12), intArrayOf(11, 12),
        intArrayOf(11, 13), intArrayOf(13, 15), intArrayOf(12, 14), intArrayOf(14, 16),
    )

    /** Push a new result and repaint. Safe to call from any thread. */
    fun setResult(result: InferenceResult, frameWidth: Int, frameHeight: Int, fpsText: String) {
        this.result = result
        this.frameWidth = frameWidth
        this.frameHeight = frameHeight
        this.fpsText = fpsText
        postInvalidate()
    }

    /**
     * Static-image test mode: draw [bitmap] as the background and overlay [result] on top, using the
     * bitmap's own dimensions as the frame size so the letterbox inverse lines up exactly.
     */
    fun setStaticResult(bitmap: Bitmap, result: InferenceResult, fpsText: String) {
        this.backgroundBitmap = bitmap
        this.result = result
        this.frameWidth = bitmap.width
        this.frameHeight = bitmap.height
        this.fpsText = fpsText
        postInvalidate()
    }

    // --- frame-space -> view-space mapping (center-crop, matches PreviewView fillCenter) ---
    private var mapScale = 1f
    private var mapDx = 0f
    private var mapDy = 0f

    private fun updateMapping() {
        if (frameWidth == 0 || frameHeight == 0) return
        // fillCenter: scale so the frame covers the view, center it (crop overflow).
        mapScale = max(width.toFloat() / frameWidth, height.toFloat() / frameHeight)
        mapDx = (width - frameWidth * mapScale) / 2f
        mapDy = (height - frameHeight * mapScale) / 2f
    }

    private fun mapX(x: Float) = x * mapScale + mapDx
    private fun mapY(y: Float) = y * mapScale + mapDy

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        updateMapping()

        // Static-image test mode: paint the source image behind the overlays, mapped the same way.
        val bg = backgroundBitmap
        if (bg != null && frameWidth > 0 && frameHeight > 0) {
            val dst = RectF(mapX(0f), mapY(0f), mapX(frameWidth.toFloat()), mapY(frameHeight.toFloat()))
            canvas.drawBitmap(bg, Rect(0, 0, bg.width, bg.height), dst, null)
        }

        // Masks first (under boxes).
        for (mask in result.masks) {
            drawMask(canvas, mask)
        }

        // Detection boxes.
        for (det in result.detections) {
            val color = colorForClass(det.classId)
            boxPaint.color = color
            val rect = RectF(mapX(det.box.left), mapY(det.box.top), mapX(det.box.right), mapY(det.box.bottom))
            canvas.drawRect(rect, boxPaint)
            drawLabel(canvas, "${det.label} ${"%.2f".format(det.score)}", rect.left, rect.top, color)
        }

        // Mask instances also get a box + label.
        for (mask in result.masks) {
            val det = mask.detection
            val color = colorForClass(det.classId)
            boxPaint.color = color
            val rect = RectF(mapX(det.box.left), mapY(det.box.top), mapX(det.box.right), mapY(det.box.bottom))
            canvas.drawRect(rect, boxPaint)
            drawLabel(canvas, "${det.label} ${"%.2f".format(det.score)}", rect.left, rect.top, color)
        }

        // Pose skeletons.
        for (pose in result.poses) {
            drawPose(canvas, pose)
        }

        // Classification: top-k list, top-left under the spinner.
        if (result.classifications.isNotEmpty()) {
            var y = 220f
            for (cls in result.classifications) {
                val line = "${cls.label}: ${"%.1f".format(cls.score * 100)}%"
                canvas.drawRect(20f, y - 34f, 20f + textPaint.measureText(line) + 16f, y + 10f, textBgPaint)
                canvas.drawText(line, 28f, y, textPaint)
                y += 52f
            }
        }

        // FPS / latency readout.
        if (fpsText.isNotEmpty()) {
            canvas.drawText(fpsText, 20f, height - 40f, fpsPaint)
        }
    }

    private fun drawLabel(canvas: Canvas, text: String, x: Float, y: Float, color: Int) {
        val w = textPaint.measureText(text) + 16f
        val top = max(0f, y - 46f)
        textBgPaint.color = Color.argb(160, Color.red(color), Color.green(color), Color.blue(color))
        canvas.drawRect(x, top, x + w, top + 46f, textBgPaint)
        canvas.drawText(text, x + 8f, top + 34f, textPaint)
    }

    private fun drawPose(canvas: Canvas, pose: Keypoints) {
        val pts = pose.points
        // Skeleton edges.
        for (edge in skeleton) {
            val a = pts.getOrNull(edge[0]) ?: continue
            val b = pts.getOrNull(edge[1]) ?: continue
            if (a.score < 0.5f || b.score < 0.5f) continue
            canvas.drawLine(mapX(a.x), mapY(a.y), mapX(b.x), mapY(b.y), skeletonPaint)
        }
        // Keypoints.
        for (p in pts) {
            if (p.score < 0.5f) continue
            canvas.drawCircle(mapX(p.x), mapY(p.y), 6f, kptPaint)
        }
        // Box + label for the person.
        val det = pose.detection
        boxPaint.color = colorForClass(det.classId)
        val rect = RectF(mapX(det.box.left), mapY(det.box.top), mapX(det.box.right), mapY(det.box.bottom))
        canvas.drawRect(rect, boxPaint)
    }

    private fun drawMask(canvas: Canvas, mask: Mask) {
        val mw = mask.maskWidth
        val mh = mask.maskHeight
        if (mw <= 0 || mh <= 0) return

        var bmp = maskBitmap
        if (bmp == null || bmp.width != mw || bmp.height != mh) {
            bmp = Bitmap.createBitmap(mw, mh, Bitmap.Config.ARGB_8888)
            maskBitmap = bmp
        }
        val color = colorForClass(mask.detection.classId)
        val pixels = IntArray(mw * mh)
        val r = Color.red(color)
        val g = Color.green(color)
        val b = Color.blue(color)
        for (i in pixels.indices) {
            // Threshold at 0.5; alpha carries the mask coverage.
            pixels[i] = if (mask.data[i] >= 0.5f) Color.argb(255, r, g, b) else Color.TRANSPARENT
        }
        bmp.setPixels(pixels, 0, mw, 0, 0, mw, mh)

        // The mask grid covers `region` in frame space; draw it there, mapped to the view.
        val dst = RectF(
            mapX(mask.region.left), mapY(mask.region.top),
            mapX(mask.region.right), mapY(mask.region.bottom),
        )
        val src = Rect(0, 0, mw, mh)
        canvas.drawBitmap(bmp, src, dst, maskPaint)
    }

    private fun colorForClass(classId: Int): Int {
        // Deterministic bright color per class id (HSV hue spread).
        val hue = (classId * 47 % 360).toFloat()
        return Color.HSVToColor(floatArrayOf(hue, 0.9f, 1.0f))
    }
}

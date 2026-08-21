// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package org.intel.openvino.demo.postprocess

/**
 * Pure-Kotlin non-maximum suppression.
 *
 * A lightweight candidate box carrying whatever index the caller wants to recover after
 * suppression (its position in the pre-NMS list). Boxes are in any consistent coordinate space.
 */
data class NmsBox(
    val left: Float,
    val top: Float,
    val right: Float,
    val bottom: Float,
    val score: Float,
    val classId: Int,
    val sourceIndex: Int,
)

object Nms {

    /**
     * Class-aware greedy NMS.
     *
     * Boxes are grouped by class so that overlapping detections of *different* classes are both
     * kept (e.g. a person standing in front of a car). Within a class, the highest-scoring box
     * suppresses lower-scoring boxes whose IoU exceeds [iouThreshold].
     *
     * @param boxes candidate boxes (already confidence-filtered by the caller)
     * @param iouThreshold IoU above which a lower-scoring box is suppressed
     * @param maxDetections cap on the number of kept boxes (safety bound)
     * @return kept boxes, highest score first
     */
    fun classAware(
        boxes: List<NmsBox>,
        iouThreshold: Float,
        maxDetections: Int = 300,
    ): List<NmsBox> {
        if (boxes.isEmpty()) return emptyList()

        val kept = ArrayList<NmsBox>()
        // Process each class independently.
        val byClass = boxes.groupBy { it.classId }
        for ((_, classBoxes) in byClass) {
            val sorted = classBoxes.sortedByDescending { it.score }
            val suppressed = BooleanArray(sorted.size)
            for (i in sorted.indices) {
                if (suppressed[i]) continue
                val a = sorted[i]
                kept.add(a)
                for (j in (i + 1) until sorted.size) {
                    if (suppressed[j]) continue
                    if (iou(a, sorted[j]) > iouThreshold) {
                        suppressed[j] = true
                    }
                }
            }
        }
        // Merge the per-class survivors, keep the strongest overall, and bound the count.
        return kept.sortedByDescending { it.score }.take(maxDetections)
    }

    /** Intersection-over-union of two boxes. */
    private fun iou(a: NmsBox, b: NmsBox): Float {
        val interLeft = maxOf(a.left, b.left)
        val interTop = maxOf(a.top, b.top)
        val interRight = minOf(a.right, b.right)
        val interBottom = minOf(a.bottom, b.bottom)

        val interW = (interRight - interLeft).coerceAtLeast(0f)
        val interH = (interBottom - interTop).coerceAtLeast(0f)
        val interArea = interW * interH
        if (interArea <= 0f) return 0f

        val areaA = (a.right - a.left).coerceAtLeast(0f) * (a.bottom - a.top).coerceAtLeast(0f)
        val areaB = (b.right - b.left).coerceAtLeast(0f) * (b.bottom - b.top).coerceAtLeast(0f)
        val union = areaA + areaB - interArea
        return if (union <= 0f) 0f else interArea / union
    }
}

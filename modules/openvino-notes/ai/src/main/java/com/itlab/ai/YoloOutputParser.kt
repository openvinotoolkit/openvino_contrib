package com.itlab.ai

internal class YoloOutputParser(
    private val config: OnDeviceVisionConfig,
) {
    fun parseTags(
        outputData: FloatArray,
        classNames: List<String>,
    ): Set<String> =
        applyNms(buildDetections(outputData))
            .mapNotNull { detection ->
                classNames.getOrNull(detection.classId) ?: "class_${detection.classId}"
            }.take(config.maxTags)
            .toSet()

    private fun buildDetections(outputData: FloatArray): List<YoloDetection> {
        val detectionCount = (outputData.size / DETECTION_STRIDE).coerceAtMost(config.maxDetections)
        return (0 until detectionCount)
            .mapNotNull { index ->
                val base = index * DETECTION_STRIDE
                val confidence = outputData[base + CONFIDENCE_OFFSET]
                if (confidence <= config.confidenceThreshold) {
                    null
                } else {
                    YoloDetection(
                        x1 = outputData[base],
                        y1 = outputData[base + 1],
                        x2 = outputData[base + 2],
                        y2 = outputData[base + 3],
                        confidence = confidence,
                        classId = outputData[base + CLASS_ID_OFFSET].toInt(),
                    )
                }
            }
    }

    private fun applyNms(detections: List<YoloDetection>): List<YoloDetection> {
        val selected = mutableListOf<YoloDetection>()
        detections
            .sortedByDescending { it.confidence }
            .forEach { candidate ->
                val overlapsSameClass =
                    selected.any {
                        it.classId == candidate.classId &&
                            calculateIou(it, candidate) > config.iouThreshold
                    }
                if (!overlapsSameClass) {
                    selected += candidate
                }
            }
        return selected
    }

    private fun calculateIou(
        first: YoloDetection,
        second: YoloDetection,
    ): Float {
        val x1 = maxOf(first.x1, second.x1)
        val y1 = maxOf(first.y1, second.y1)
        val x2 = minOf(first.x2, second.x2)
        val y2 = minOf(first.y2, second.y2)
        val intersectionWidth = (x2 - x1).coerceAtLeast(0f)
        val intersectionHeight = (y2 - y1).coerceAtLeast(0f)
        val intersection = intersectionWidth * intersectionHeight
        val union = first.area + second.area - intersection
        return if (union > 0f) intersection / union else 0f
    }

    private companion object {
        const val DETECTION_STRIDE = 6
        const val CONFIDENCE_OFFSET = 4
        const val CLASS_ID_OFFSET = 5
    }
}

internal data class YoloDetection(
    val x1: Float,
    val y1: Float,
    val x2: Float,
    val y2: Float,
    val confidence: Float,
    val classId: Int,
) {
    val area: Float
        get() = (x2 - x1).coerceAtLeast(0f) * (y2 - y1).coerceAtLeast(0f)
}

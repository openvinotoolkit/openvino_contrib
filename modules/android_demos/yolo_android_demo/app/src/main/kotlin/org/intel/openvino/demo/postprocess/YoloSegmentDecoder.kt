// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package org.intel.openvino.demo.postprocess

import org.intel.openvino.demo.inference.ModelConfig
import org.intel.openvino.demo.preprocess.LetterboxTransform
import kotlin.math.exp

/**
 * Anchor-free YOLOv8/YOLO11 instance segmentation decoder.
 *
 * Two outputs:
 *  - detections `[1, 4 + numClasses + numMasks, numBoxes]` (e.g. `[1, 116, 8400]`, numMasks=32),
 *  - mask prototypes `[1, numMasks, mh, mw]` (e.g. `[1, 32, 160, 160]`).
 *
 * After confidence filtering + NMS, each kept detection's mask coefficients are matrix-multiplied
 * by the prototypes, passed through a sigmoid, cropped to the detection box (in prototype space),
 * and emitted as a small probability grid. The overlay upsamples and draws it translucently.
 */
class YoloSegmentDecoder : YoloDecoder {

    override fun decode(
        outputs: List<RawOutput>,
        transform: LetterboxTransform,
        config: ModelConfig,
        labels: List<String>,
        confThreshold: Float,
        iouThreshold: Float,
    ): InferenceResult {
        // Output 0 is the detection head, output 1 the prototypes. The order matches the manifest
        // (index 0 = [1,116,8400], index 1 = [1,32,160,160]); guard by rank in case it flips.
        val detOut: RawOutput
        val protoOut: RawOutput
        if (outputs[0].shape.size == 3) {
            detOut = outputs[0]; protoOut = outputs[1]
        } else {
            detOut = outputs[1]; protoOut = outputs[0]
        }

        val channels = detOut.shape[1]
        val numBoxes = detOut.shape[2]
        val numMasks = protoOut.shape[1]
        val protoH = protoOut.shape[2]
        val protoW = protoOut.shape[3]
        val numClasses = channels - 4 - numMasks

        val candidates = DetectHead.collect(detOut.data, channels, numBoxes, numClasses, confThreshold)

        val nmsInput = candidates.mapIndexed { i, c ->
            val box = DetectHead.toFrameBox(c, transform)
            NmsBox(box.left, box.top, box.right, box.bottom, c.score, c.classId, i)
        }
        val kept = Nms.classAware(nmsInput, iouThreshold, maxDetections = 64)

        val coeffBase = (4 + numClasses) * numBoxes
        val protoPlane = protoH * protoW
        // Model input size, to map prototype grid <-> model pixels <-> frame pixels.
        val inputW = config.inputWidth.toFloat()
        val inputH = config.inputHeight.toFloat()

        val masks = ArrayList<Mask>(kept.size)
        for (nms in kept) {
            val c = candidates[nms.sourceIndex]
            val anchor = c.anchorIndex

            // Read this detection's mask coefficients.
            val coeffs = FloatArray(numMasks) { m -> detOut.data[coeffBase + m * numBoxes + anchor] }

            // mask(y,x) = sigmoid( sum_m coeffs[m] * proto[m](y,x) ), over the prototype grid.
            val maskData = FloatArray(protoPlane)
            for (m in 0 until numMasks) {
                val coeff = coeffs[m]
                if (coeff == 0f) continue
                val base = m * protoPlane
                var p = 0
                while (p < protoPlane) {
                    maskData[p] += coeff * protoOut.data[base + p]
                    p++
                }
            }
            for (p in 0 until protoPlane) {
                maskData[p] = sigmoid(maskData[p])
            }

            // Detection box in model-input pixels, then to prototype-grid coordinates for cropping.
            val protoScaleX = protoW / inputW
            val protoScaleY = protoH / inputH
            val modelLeft = (c.cx - c.w / 2f)
            val modelTop = (c.cy - c.h / 2f)
            val modelRight = (c.cx + c.w / 2f)
            val modelBottom = (c.cy + c.h / 2f)
            val gx0 = (modelLeft * protoScaleX).toInt().coerceIn(0, protoW - 1)
            val gy0 = (modelTop * protoScaleY).toInt().coerceIn(0, protoH - 1)
            val gx1 = (modelRight * protoScaleX).toInt().coerceIn(0, protoW - 1)
            val gy1 = (modelBottom * protoScaleY).toInt().coerceIn(0, protoH - 1)

            // Zero out everything outside the detection box (crop-to-box, YOLO convention).
            for (y in 0 until protoH) {
                val inRowBox = y in gy0..gy1
                val rowBase = y * protoW
                for (x in 0 until protoW) {
                    if (!inRowBox || x < gx0 || x > gx1) {
                        maskData[rowBase + x] = 0f
                    }
                }
            }

            // The prototype grid covers the full letterboxed model input; report the frame region
            // that the *whole grid* maps to, so the overlay can place it with the inverse transform.
            val region = Box(
                left = transform.frameX(0f),
                top = transform.frameY(0f),
                right = transform.frameX(inputW),
                bottom = transform.frameY(inputH),
            )

            masks.add(
                Mask(
                    detection = Detection(
                        box = Box(nms.left, nms.top, nms.right, nms.bottom),
                        classId = nms.classId,
                        score = nms.score,
                        label = DetectHead.labelFor(labels, nms.classId),
                    ),
                    data = maskData,
                    maskWidth = protoW,
                    maskHeight = protoH,
                    region = region,
                )
            )
        }
        return InferenceResult(masks = masks)
    }

    private fun sigmoid(x: Float): Float = (1.0 / (1.0 + exp(-x.toDouble()))).toFloat()
}

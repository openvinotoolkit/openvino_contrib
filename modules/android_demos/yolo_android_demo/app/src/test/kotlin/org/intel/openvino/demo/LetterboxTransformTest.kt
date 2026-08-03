// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package org.intel.openvino.demo

import org.intel.openvino.demo.preprocess.LetterboxTransform
import org.junit.Assert.assertEquals
import org.junit.Test

/**
 * Verifies the letterbox forward/inverse math on a known image. A wrong inverse letterbox is the
 * single most common source of "boxes are shifted" bugs, so it is pinned here.
 */
class LetterboxTransformTest {

    @Test
    fun landscapeFitsWidthAndPadsHeight() {
        // 1920x1080 frame into a 640x640 square: scale = 640/1920 = 0.3333...,
        // scaled height = 1080 * 0.3333 = 360, vertical pad = (640 - 360)/2 = 140.
        val t = LetterboxTransform.compute(1920, 1080, 640, 640)
        assertEquals(640f / 1920f, t.scale, 1e-6f)
        assertEquals(0f, t.padX, 1e-4f)
        assertEquals(140f, t.padY, 1e-3f)
    }

    @Test
    fun portraitFitsHeightAndPadsWidth() {
        // 1080x1920 into 640x640: scale = 640/1920, scaled width = 360, horizontal pad = 140.
        val t = LetterboxTransform.compute(1080, 1920, 640, 640)
        assertEquals(640f / 1920f, t.scale, 1e-6f)
        assertEquals(140f, t.padX, 1e-3f)
        assertEquals(0f, t.padY, 1e-4f)
    }

    @Test
    fun inverseMapsModelCornersBackToFrameCorners() {
        val frameW = 1280
        val frameH = 720
        val t = LetterboxTransform.compute(frameW, frameH, 640, 640)

        // The frame's top-left (0,0) maps to model (padX, padY); invert back to (0,0).
        assertEquals(0f, t.frameX(t.padX), 1e-3f)
        assertEquals(0f, t.frameY(t.padY), 1e-3f)

        // The frame's bottom-right maps to (padX + w*scale, padY + h*scale); invert back.
        val modelRightX = t.padX + frameW * t.scale
        val modelBottomY = t.padY + frameH * t.scale
        assertEquals(frameW.toFloat(), t.frameX(modelRightX), 1e-2f)
        assertEquals(frameH.toFloat(), t.frameY(modelBottomY), 1e-2f)
    }

    @Test
    fun roundTripKnownPoint() {
        val t = LetterboxTransform.compute(1920, 1080, 640, 640)
        // A known frame point -> model space -> back must be identity.
        val fx = 960f // frame center x
        val fy = 540f // frame center y
        val modelX = fx * t.scale + t.padX
        val modelY = fy * t.scale + t.padY
        assertEquals(fx, t.frameX(modelX), 1e-2f)
        assertEquals(fy, t.frameY(modelY), 1e-2f)
        // Model center should be 320,320 for a centered frame.
        assertEquals(320f, modelX, 1e-2f)
        assertEquals(320f, modelY, 1e-2f)
    }
}

// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package org.intel.openvino.demo

import org.intel.openvino.demo.postprocess.Nms
import org.intel.openvino.demo.postprocess.NmsBox
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/** Pure-Kotlin NMS behavior (class-aware), replacing OpenCV's Dnn.NMSBoxes. */
class NmsTest {

    @Test
    fun suppressesOverlappingSameClass() {
        val boxes = listOf(
            NmsBox(0f, 0f, 100f, 100f, 0.9f, classId = 0, sourceIndex = 0),
            NmsBox(10f, 10f, 105f, 105f, 0.8f, classId = 0, sourceIndex = 1), // high IoU with #0
        )
        val kept = Nms.classAware(boxes, iouThreshold = 0.5f)
        assertEquals(1, kept.size)
        assertEquals(0.9f, kept[0].score, 1e-6f) // the stronger box survives
    }

    @Test
    fun keepsOverlappingDifferentClasses() {
        val boxes = listOf(
            NmsBox(0f, 0f, 100f, 100f, 0.9f, classId = 0, sourceIndex = 0),
            NmsBox(5f, 5f, 100f, 100f, 0.85f, classId = 1, sourceIndex = 1), // overlaps but other class
        )
        val kept = Nms.classAware(boxes, iouThreshold = 0.5f)
        assertEquals(2, kept.size) // class-aware: both kept
    }

    @Test
    fun keepsDistantSameClass() {
        val boxes = listOf(
            NmsBox(0f, 0f, 50f, 50f, 0.9f, classId = 0, sourceIndex = 0),
            NmsBox(200f, 200f, 260f, 260f, 0.7f, classId = 0, sourceIndex = 1), // no overlap
        )
        val kept = Nms.classAware(boxes, iouThreshold = 0.5f)
        assertEquals(2, kept.size)
    }

    @Test
    fun resultsSortedByScoreDescending() {
        val boxes = listOf(
            NmsBox(0f, 0f, 10f, 10f, 0.3f, classId = 0, sourceIndex = 0),
            NmsBox(100f, 100f, 110f, 110f, 0.95f, classId = 1, sourceIndex = 1),
            NmsBox(200f, 200f, 210f, 210f, 0.6f, classId = 2, sourceIndex = 2),
        )
        val kept = Nms.classAware(boxes, iouThreshold = 0.5f)
        assertEquals(3, kept.size)
        assertTrue(kept[0].score >= kept[1].score)
        assertTrue(kept[1].score >= kept[2].score)
    }

    @Test
    fun emptyInputYieldsEmpty() {
        assertEquals(0, Nms.classAware(emptyList(), 0.5f).size)
    }
}

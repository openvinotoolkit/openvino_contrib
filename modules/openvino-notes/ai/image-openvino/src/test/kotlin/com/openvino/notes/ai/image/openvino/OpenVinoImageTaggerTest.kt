// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package com.openvino.notes.ai.image.openvino

import com.openvino.notes.ai.image.api.ImageInput
import com.openvino.notes.ai.image.api.ImageOutcome
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Test

class OpenVinoImageTaggerTest {
    @Test fun `invalid input is rejected before unavailable model mapping`() = runTest {
        val tagger = OpenVinoImageTagger(null)

        assertEquals(
            ImageOutcome.Failed("image_openvino.invalid_input"),
            tagger.tag(ImageInput.Bytes(byteArrayOf(), "image/jpeg")),
        )
        assertEquals(
            ImageOutcome.NotConfigured("Image model is not loaded"),
            tagger.tag(ImageInput.Bytes(byteArrayOf(1), "image/jpeg")),
        )
    }
}

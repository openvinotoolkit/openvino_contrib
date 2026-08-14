// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package com.itlab.ai.image.openvino

import com.itlab.ai.image.api.ImageInput
import com.itlab.ai.image.api.ImageModelState
import com.itlab.ai.image.api.ImageOutcome
import com.itlab.ai.image.api.ImageTagger
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow

internal class OpenVinoImageTagger(private val modelDirectory: String?) : ImageTagger {
    private val mutableState = MutableStateFlow(ImageModelState.UNAVAILABLE)
    override val state: StateFlow<ImageModelState> = mutableState

    override suspend fun load(): ImageOutcome {
        mutableState.value = ImageModelState.UNAVAILABLE
        return ImageOutcome.NotConfigured(modelDirectory?.let { "OpenVINO runtime artifact is not connected" } ?: "Image model directory is not configured")
    }

    override suspend fun tag(input: ImageInput): ImageOutcome {
        if (input is ImageInput.Bytes && (input.value.isEmpty() || !input.mediaType.startsWith("image/"))) {
            return ImageOutcome.Failed("image_openvino.invalid_input")
        }
        return when (state.value) {
            ImageModelState.READY -> ImageOutcome.Failed("image_openvino.native_bridge_missing")
            else -> ImageOutcome.NotConfigured("Image model is not loaded")
        }
    }

    override fun close() { mutableState.value = ImageModelState.CLOSED }
}

data class OpenVinoImageComponent(val tagger: ImageTagger) {
    companion object {
        fun create(modelDirectory: String? = null): OpenVinoImageComponent = OpenVinoImageComponent(OpenVinoImageTagger(modelDirectory))
    }
}

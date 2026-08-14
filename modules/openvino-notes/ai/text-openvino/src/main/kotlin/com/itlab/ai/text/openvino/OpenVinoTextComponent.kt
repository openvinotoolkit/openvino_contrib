// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package com.itlab.ai.text.openvino

import com.itlab.ai.text.api.TextAssistant
import com.itlab.ai.text.api.TextModelState
import com.itlab.ai.text.api.TextOutcome
import com.itlab.ai.text.api.TextRequest
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow

internal class OpenVinoTextAssistant(private val modelDirectory: String?) : TextAssistant {
    private val mutableState = MutableStateFlow(TextModelState.UNAVAILABLE)
    override val state: StateFlow<TextModelState> = mutableState

    override suspend fun load(): TextOutcome {
        mutableState.value = TextModelState.UNAVAILABLE
        return TextOutcome.NotConfigured(modelDirectory?.let { "OpenVINO runtime artifact is not connected" } ?: "Text model directory is not configured")
    }

    override suspend fun generate(request: TextRequest): TextOutcome = when (state.value) {
        TextModelState.READY -> TextOutcome.Failed("text_openvino.native_bridge_missing")
        else -> TextOutcome.NotConfigured("Text model is not loaded")
    }

    override fun close() { mutableState.value = TextModelState.CLOSED }
}

data class OpenVinoTextComponent(val assistant: TextAssistant) {
    companion object {
        fun create(modelDirectory: String? = null): OpenVinoTextComponent = OpenVinoTextComponent(OpenVinoTextAssistant(modelDirectory))
    }
}

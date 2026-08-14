// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package com.itlab.ai.image.api

import kotlinx.coroutines.flow.StateFlow

sealed interface ImageInput {
    data class Bytes(val value: ByteArray, val mediaType: String) : ImageInput
}

data class ImageTag(val label: String, val confidence: Float)
enum class ImageModelState { UNAVAILABLE, LOADING, READY, FAILED, CLOSED }

sealed interface ImageOutcome {
    data class Tagged(val tags: List<ImageTag>) : ImageOutcome
    data class NotConfigured(val reason: String) : ImageOutcome
    data class Failed(val code: String) : ImageOutcome
}

interface ImageTagger : AutoCloseable {
    val state: StateFlow<ImageModelState>
    suspend fun load(): ImageOutcome
    suspend fun tag(input: ImageInput): ImageOutcome
    override fun close()
}

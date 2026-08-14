// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package com.itlab.ai.image.api

import kotlinx.coroutines.flow.MutableStateFlow

class FakeImageTagger : ImageTagger {
    override val state = MutableStateFlow(ImageModelState.READY)
    var outcome: ImageOutcome = ImageOutcome.Tagged(emptyList())
    override suspend fun load(): ImageOutcome = outcome
    override suspend fun tag(input: ImageInput): ImageOutcome = outcome
    override fun close() { state.value = ImageModelState.CLOSED }
}

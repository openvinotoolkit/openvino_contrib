// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package com.itlab.ai.text.api

import kotlinx.coroutines.flow.MutableStateFlow

class FakeTextAssistant : TextAssistant {
    override val state = MutableStateFlow(TextModelState.READY)
    var outcome: TextOutcome = TextOutcome.Generated("fake response")
    override suspend fun load(): TextOutcome = outcome
    override suspend fun generate(request: TextRequest): TextOutcome = outcome
    override fun close() { state.value = TextModelState.CLOSED }
}

// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package com.itlab.ai.text.api

import kotlinx.coroutines.flow.MutableStateFlow

class FakeTextAssistant : TextAssistant {
    override val state = MutableStateFlow(TextModelState.READY)
    var loadOutcome: ModelLoadOutcome = ModelLoadOutcome.Ready
    var summaryOutcome: SummaryOutcome = SummaryOutcome.Generated("fake summary")
    var tagsOutcome: TextTagsOutcome = TextTagsOutcome.Generated(setOf("fake"))
    var rewriteOutcome: RewriteOutcome = RewriteOutcome.Generated("fake rewrite")

    override suspend fun load(): ModelLoadOutcome = loadOutcome
    override suspend fun summarize(request: SummaryRequest): SummaryOutcome = summaryOutcome
    override suspend fun suggestTags(request: TextTagsRequest): TextTagsOutcome = tagsOutcome
    override suspend fun rewrite(request: RewriteRequest): RewriteOutcome = rewriteOutcome
    override fun close() { state.value = TextModelState.CLOSED }
}

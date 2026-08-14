// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package com.openvino.notes.ai.text.api

import kotlinx.coroutines.flow.StateFlow

enum class TextModelState { UNAVAILABLE, LOADING, READY, FAILED, CLOSED }
enum class RewriteStyle { CONCISE, CLEAR, PROFESSIONAL, CASUAL }

@JvmInline
value class LanguageHint(val bcp47Tag: String) {
    init { require(bcp47Tag.isNotBlank()) { "Language hint must not be blank" } }
}

data class SummaryRequest(
    val content: String,
    val maxCharacters: Int = 2_000,
    val languageHint: LanguageHint? = null,
)
data class TextTagsRequest(
    val content: String,
    val maxTags: Int = 8,
    val languageHint: LanguageHint? = null,
)
data class RewriteRequest(
    val content: String,
    val style: RewriteStyle,
    val maxCharacters: Int = 10_000,
    val languageHint: LanguageHint? = null,
)

sealed interface ModelLoadOutcome {
    data object Ready : ModelLoadOutcome
    data class NotConfigured(val reason: String) : ModelLoadOutcome
    data class Failed(val code: String) : ModelLoadOutcome
}

sealed interface SummaryOutcome {
    data class Generated(val summary: String) : SummaryOutcome
    data class NotConfigured(val reason: String) : SummaryOutcome
    data class Failed(val code: String) : SummaryOutcome
}

sealed interface TextTagsOutcome {
    data class Generated(val tags: Set<String>) : TextTagsOutcome
    data class NotConfigured(val reason: String) : TextTagsOutcome
    data class Failed(val code: String) : TextTagsOutcome
}

sealed interface RewriteOutcome {
    data class Generated(val content: String) : RewriteOutcome
    data class NotConfigured(val reason: String) : RewriteOutcome
    data class Failed(val code: String) : RewriteOutcome
}

interface TextAssistant : AutoCloseable {
    val state: StateFlow<TextModelState>
    suspend fun load(): ModelLoadOutcome
    suspend fun summarize(request: SummaryRequest): SummaryOutcome
    suspend fun suggestTags(request: TextTagsRequest): TextTagsOutcome
    suspend fun rewrite(request: RewriteRequest): RewriteOutcome
    override fun close()
}

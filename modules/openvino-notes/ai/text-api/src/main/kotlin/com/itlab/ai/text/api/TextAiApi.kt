package com.itlab.ai.text.api

import kotlinx.coroutines.flow.StateFlow

enum class TextModelState { UNAVAILABLE, LOADING, READY, FAILED, CLOSED }
data class TextRequest(val prompt: String, val context: String, val maxOutputCharacters: Int = 2_000)

sealed interface TextOutcome {
    data class Generated(val text: String) : TextOutcome
    data class NotConfigured(val reason: String) : TextOutcome
    data class Failed(val code: String) : TextOutcome
}

interface TextAssistant : AutoCloseable {
    val state: StateFlow<TextModelState>
    suspend fun load(): TextOutcome
    suspend fun generate(request: TextRequest): TextOutcome
    override fun close()
}

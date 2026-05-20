package com.itlab.ai

interface LlmInferenceBackend {
    fun generate(
        prompt: String,
        maxNewTokens: Int,
        intent: LlmGenerationIntent = LlmGenerationIntent.General,
    ): String

    fun warmUp() = Unit

    fun release() = Unit
}

enum class LlmGenerationIntent {
    General,
    Summary,
    Tags,
    Rewrite,
}

class MissingLlmRuntimeException(
    message: String,
    cause: Throwable? = null,
) : IllegalStateException(message, cause)

class UnavailableLlmBackend(
    private val reason: String = "OpenVINO GenAI backend is not configured.",
) : LlmInferenceBackend {
    override fun generate(
        prompt: String,
        maxNewTokens: Int,
        intent: LlmGenerationIntent,
    ): String = throw MissingLlmRuntimeException(reason)
}

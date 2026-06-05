package com.itlab.domain.ai

interface NoteAiService {
    suspend fun warmUp() = Unit

    fun release() = Unit

    suspend fun summarize(
        text: String,
        maxInputTokens: Int = 512,
        maxNewTokens: Int = 128,
    ): String

    suspend fun suggestTags(
        text: String,
        maxInputTokens: Int = 384,
        maxTags: Int = 4,
    ): Set<String>

    suspend fun rewrite(
        text: String,
        style: RewriteStyle,
        maxInputTokens: Int = 768,
        maxNewTokens: Int = 192,
    ): String

    suspend fun tagIMGs(img: List<String>): Set<String>

    @Deprecated("Use suggestTags with explicit LLM limits.")
    suspend fun tagTXT(text: String): Set<String> = suggestTags(text)
}

enum class RewriteStyle {
    CLEANUP,
}

package com.itlab.ai

import com.itlab.domain.ai.RewriteStyle

class OpenVinoEngine(
    private val llmBackend: LlmInferenceBackend = UnavailableLlmBackend(),
    private val promptBuilder: NoteLlmPromptBuilder = NoteLlmPromptBuilder(),
    private val config: OnDeviceLlmConfig = OnDeviceLlmConfig.defaultAndroid(),
) {
    fun warmUp() {
        llmBackend.warmUp()
    }

    fun release() {
        llmBackend.release()
    }

    fun runLlmSummary(
        text: String,
        maxInputTokens: Int = config.summaryMaxInputTokens,
        maxNewTokens: Int = config.summaryMaxNewTokens,
    ): String {
        if (text.isBlank()) return ""
        return llmBackend.generate(
            prompt = promptBuilder.summaryPrompt(text, maxInputTokens),
            maxNewTokens = maxNewTokens,
            intent = LlmGenerationIntent.Summary,
        )
    }

    fun runLlmSummaryRetry(
        text: String,
        previousAnswer: String,
        maxInputTokens: Int = config.summaryMaxInputTokens,
        maxNewTokens: Int = config.summaryMaxNewTokens,
    ): String {
        if (text.isBlank()) return ""
        return llmBackend.generate(
            prompt = promptBuilder.summaryRetryPrompt(text, previousAnswer, maxInputTokens),
            maxNewTokens = maxNewTokens,
            intent = LlmGenerationIntent.Summary,
        )
    }

    fun runLlmTagging(
        text: String,
        maxInputTokens: Int = config.tagsMaxInputTokens,
        maxNewTokens: Int = config.tagsMaxNewTokens,
        maxTags: Int = config.maxTags,
    ): String {
        if (text.isBlank()) return ""
        return llmBackend.generate(
            prompt = promptBuilder.tagsPrompt(text, maxInputTokens, maxTags),
            maxNewTokens = maxNewTokens,
            intent = LlmGenerationIntent.Tags,
        )
    }

    fun runLlmTaggingRetry(
        text: String,
        previousAnswer: String,
        maxInputTokens: Int = config.tagsMaxInputTokens,
        maxNewTokens: Int = config.tagsMaxNewTokens,
        maxTags: Int = config.maxTags,
    ): String {
        if (text.isBlank()) return ""
        return llmBackend.generate(
            prompt = promptBuilder.tagsRetryPrompt(text, previousAnswer, maxInputTokens, maxTags),
            maxNewTokens = maxNewTokens,
            intent = LlmGenerationIntent.Tags,
        )
    }

    fun runLlmRewrite(
        text: String,
        style: RewriteStyle,
        maxInputTokens: Int = config.rewriteMaxInputTokens,
        maxNewTokens: Int = config.rewriteMaxNewTokens,
    ): String {
        if (text.isBlank()) return ""
        return llmBackend.generate(
            prompt = promptBuilder.rewritePrompt(text, style, maxInputTokens),
            maxNewTokens = maxNewTokens,
            intent = LlmGenerationIntent.Rewrite,
        )
    }

    fun runLlmRewriteRetry(
        text: String,
        style: RewriteStyle,
        previousAnswer: String,
        maxInputTokens: Int = config.rewriteMaxInputTokens,
        maxNewTokens: Int = config.rewriteMaxNewTokens,
    ): String {
        if (text.isBlank()) return ""
        return llmBackend.generate(
            prompt = promptBuilder.rewriteRetryPrompt(text, style, previousAnswer, maxInputTokens),
            maxNewTokens = maxNewTokens,
            intent = LlmGenerationIntent.Rewrite,
        )
    }
}

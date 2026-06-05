package com.itlab.ai

import android.annotation.SuppressLint
import android.util.Log
import com.ovx.openvino.genai.GenerationPerfMetrics
import com.ovx.openvino.genai.GenerationResult

internal data class GenerationLogContext(
    val intent: LlmGenerationIntent,
    val coldPipeline: Boolean,
    val maxNewTokens: Int,
    val promptChars: Int,
    val requestElapsedMs: Long,
    val generationElapsedMs: Long,
)

internal fun shouldStopEarly(
    intent: LlmGenerationIntent,
    response: String,
    maxTags: Int,
): Boolean {
    val visible = stripReasoningSections(response).trim()
    if (visible.isEmpty()) {
        return false
    }
    return when (intent) {
        LlmGenerationIntent.Summary -> hasCompleteSummary(visible)
        LlmGenerationIntent.Tags -> hasEnoughTags(visible, maxTags)
        LlmGenerationIntent.Rewrite -> hasCompleteRewrite(visible)
        LlmGenerationIntent.General -> false
    }
}

@SuppressLint("LogConditional")
internal fun logGeneration(
    context: GenerationLogContext,
    result: GenerationResult,
) {
    Log.i(
        "OpenVinoGenAiBackend",
        "generate intent=${context.intent} coldPipeline=${context.coldPipeline} status=${result.status()} " +
            "promptChars=${context.promptChars} maxNewTokens=${context.maxNewTokens} " +
            "requestMs=${context.requestElapsedMs} generationMs=${context.generationElapsedMs} " +
            "metrics=${GenerationPerfMetrics.from(result).compactString()}",
    )
}

internal fun preparePrompt(
    prompt: String,
    config: OnDeviceLlmConfig,
): String {
    val hint = config.disableReasoningPromptHint.trim()
    val trimmedPrompt = prompt.trimEnd()
    val shouldAppendHint =
        !config.includeReasoningOutput &&
            prompt.isNotBlank() &&
            hint.isNotEmpty() &&
            !trimmedPrompt.endsWith(hint, ignoreCase = true)

    return if (shouldAppendHint) {
        "$trimmedPrompt\n$hint"
    } else {
        prompt
    }
}

internal fun stripReasoningSections(response: String): String {
    if (response.isBlank()) {
        return response
    }

    return THINKING_TAG_REGEX
        .replace(THINKING_BLOCK_REGEX.replace(response, ""), "")
        .trim()
}

private fun hasCompleteSummary(response: String): Boolean {
    val firstLine = response.firstVisibleLine()
    return firstLine.length >= MIN_SUMMARY_CHARS && SUMMARY_END_REGEX.containsMatchIn(firstLine)
}

private fun hasEnoughTags(
    response: String,
    maxTags: Int,
): Boolean {
    val tags =
        response
            .firstVisibleLine()
            .split(',')
            .map { it.trim() }
            .filter { it.length >= MIN_TAG_CHARS }
    return tags.size >= maxTags
}

private fun hasCompleteRewrite(response: String): Boolean {
    val text = response.trim()
    return when {
        text.length >= MAX_REWRITE_CHARS -> true
        text.length < MIN_REWRITE_CHARS -> false
        else -> REWRITE_END_REGEX.containsMatchIn(text)
    }
}

private fun String.firstVisibleLine(): String =
    lineSequence()
        .firstOrNull()
        ?.trim()
        .orEmpty()

private const val MIN_SUMMARY_CHARS = 24
private const val MIN_TAG_CHARS = 2
private const val MIN_REWRITE_CHARS = 24
private const val MAX_REWRITE_CHARS = 360
private val SUMMARY_END_REGEX = Regex("""[.!?](?:\s|$)""")
private val REWRITE_END_REGEX = Regex("""[.!?…](?:\s|$)""")
private val THINKING_BLOCK_REGEX =
    Regex(
        pattern = "<think>.*?</think>",
        options = setOf(RegexOption.IGNORE_CASE, RegexOption.DOT_MATCHES_ALL),
    )
private val THINKING_TAG_REGEX = Regex("</?think>", RegexOption.IGNORE_CASE)

// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package com.openvino.notes.ai.text.openvino

import com.openvino.notes.ai.text.api.ModelLoadOutcome
import com.openvino.notes.ai.text.api.RewriteOutcome
import com.openvino.notes.ai.text.api.RewriteRequest
import com.openvino.notes.ai.text.api.SummaryOutcome
import com.openvino.notes.ai.text.api.SummaryRequest
import com.openvino.notes.ai.text.api.TextAssistant
import com.openvino.notes.ai.text.api.TextModelState
import com.openvino.notes.ai.text.api.TextTagsOutcome
import com.openvino.notes.ai.text.api.TextTagsRequest
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow

internal fun interface TextGenerationBackend {
    suspend fun generate(input: String, maxCharacters: Int): String
}

internal class TextGenerationException(
    val diagnosticCode: String,
    val retryable: Boolean,
    cause: Throwable? = null,
) : RuntimeException(diagnosticCode, cause)

private class SummaryPipeline(private val generator: RetryingGenerator) {
    suspend fun run(request: SummaryRequest): String = generator.generate(
        instruction = "Summarize the note faithfully. Return only the summary.${request.languageInstruction()}",
        content = request.content,
        maxCharacters = request.maxCharacters,
    ).normalizeText("summary")
}

private class TextTagsPipeline(private val generator: RetryingGenerator) {
    suspend fun run(request: TextTagsRequest): Set<String> = generator.generate(
        instruction = "Suggest at most ${request.maxTags} short tags. Return comma-separated tags only.${request.languageInstruction()}",
        content = request.content,
        maxCharacters = 500,
    ).removePrefixCaseInsensitive("tags:")
        .split(',', '\n')
        .map(String::trim)
        .map { it.trimStart('-', '*', '•').trim().removePrefix("#").lowercase() }
        .filter(String::isNotBlank)
        .take(request.maxTags)
        .toSet()
}

private class RewritePipeline(private val generator: RetryingGenerator) {
    suspend fun run(request: RewriteRequest): String = generator.generate(
        instruction = "Rewrite the text in a ${request.style.name.lowercase()} style. Return only the rewritten text.${request.languageInstruction()}",
        content = request.content,
        maxCharacters = request.maxCharacters,
    ).normalizeText("rewritten text")
}

internal class RetryingGenerator(
    private val backend: TextGenerationBackend,
    private val maxAttempts: Int = 2,
) {
    suspend fun generate(instruction: String, content: String, maxCharacters: Int): String {
        var lastFailure: Throwable? = null
        repeat(maxAttempts) {
            try {
                val output = backend.generate("$instruction\n\nCONTENT:\n$content", maxCharacters).trim()
                if (output.isNotEmpty()) return output.take(maxCharacters)
                lastFailure = IllegalStateException("empty model output")
            } catch (cancelled: CancellationException) {
                throw cancelled
            } catch (failure: TextGenerationException) {
                lastFailure = failure
                if (!failure.retryable) throw failure
            } catch (failure: Throwable) {
                throw TextGenerationException("text_openvino.generation_failed", retryable = false, failure)
            }
        }
        throw requireNotNull(lastFailure)
    }
}

internal class OpenVinoTextAssistant(
    private val modelDirectory: String?,
    backend: TextGenerationBackend? = null,
) : TextAssistant {
    private val mutableState = MutableStateFlow(if (backend == null) TextModelState.UNAVAILABLE else TextModelState.READY)
    override val state: StateFlow<TextModelState> = mutableState
    private val generator = backend?.let(::RetryingGenerator)
    private val summaryPipeline = generator?.let(::SummaryPipeline)
    private val tagsPipeline = generator?.let(::TextTagsPipeline)
    private val rewritePipeline = generator?.let(::RewritePipeline)

    override suspend fun load(): ModelLoadOutcome = if (generator == null) {
        mutableState.value = TextModelState.UNAVAILABLE
        ModelLoadOutcome.NotConfigured(
            modelDirectory?.let { "OpenVINO runtime artifact is not connected" }
                ?: "Text model directory is not configured",
        )
    } else {
        mutableState.value = TextModelState.READY
        ModelLoadOutcome.Ready
    }

    override suspend fun summarize(request: SummaryRequest): SummaryOutcome = runPipeline(
        block = { requireNotNull(summaryPipeline).run(request) },
        success = SummaryOutcome::Generated,
        notConfigured = SummaryOutcome::NotConfigured,
        failed = SummaryOutcome::Failed,
    )

    override suspend fun suggestTags(request: TextTagsRequest): TextTagsOutcome = runPipeline(
        block = { requireNotNull(tagsPipeline).run(request) },
        success = TextTagsOutcome::Generated,
        notConfigured = TextTagsOutcome::NotConfigured,
        failed = TextTagsOutcome::Failed,
    )

    override suspend fun rewrite(request: RewriteRequest): RewriteOutcome = runPipeline(
        block = { requireNotNull(rewritePipeline).run(request) },
        success = RewriteOutcome::Generated,
        notConfigured = RewriteOutcome::NotConfigured,
        failed = RewriteOutcome::Failed,
    )

    private suspend fun <T, O> runPipeline(
        block: suspend () -> T,
        success: (T) -> O,
        notConfigured: (String) -> O,
        failed: (String) -> O,
    ): O = when (state.value) {
        TextModelState.READY -> try {
            success(block())
        } catch (cancelled: CancellationException) {
            throw cancelled
        } catch (failure: TextGenerationException) {
            failed(failure.diagnosticCode)
        } catch (_: Throwable) {
            failed("text_openvino.generation_failed")
        }
        else -> notConfigured("Text model is not loaded")
    }

    override fun close() { mutableState.value = TextModelState.CLOSED }
}

private fun String.normalizeText(prefix: String): String = trim()
    .removePrefixCaseInsensitive("$prefix:")
    .removePrefixCaseInsensitive("answer:")
    .trim()

private fun String.removePrefixCaseInsensitive(prefix: String): String =
    if (startsWith(prefix, ignoreCase = true)) drop(prefix.length).trimStart() else this

private fun SummaryRequest.languageInstruction(): String = languageHint?.let { " Use language ${it.bcp47Tag}." }.orEmpty()
private fun TextTagsRequest.languageInstruction(): String = languageHint?.let { " Use language ${it.bcp47Tag}." }.orEmpty()
private fun RewriteRequest.languageInstruction(): String = languageHint?.let { " Use language ${it.bcp47Tag}." }.orEmpty()

data class OpenVinoTextComponent(val assistant: TextAssistant) {
    companion object {
        fun create(modelDirectory: String? = null): OpenVinoTextComponent =
            OpenVinoTextComponent(OpenVinoTextAssistant(modelDirectory))
    }
}

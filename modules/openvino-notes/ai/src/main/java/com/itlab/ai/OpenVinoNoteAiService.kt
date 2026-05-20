package com.itlab.ai

import com.itlab.domain.ai.NoteAiService
import com.itlab.domain.ai.RewriteStyle
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

class OpenVinoNoteAiService(
    private val engine: OpenVinoEngine,
    private val processor: ResultProcessor,
    private val imageTaggingBackend: ImageTaggingBackend = UnavailableImageTaggingBackend(),
) : NoteAiService {
    override suspend fun warmUp() {
        withContext(Dispatchers.Default) {
            engine.warmUp()
        }
    }

    override fun release() {
        engine.release()
        imageTaggingBackend.release()
    }

    override suspend fun summarize(
        text: String,
        maxInputTokens: Int,
        maxNewTokens: Int,
    ): String =
        withContext(Dispatchers.Default) {
            val firstResult = engine.runLlmSummary(text, maxInputTokens, maxNewTokens)
            val llmResult =
                if (processor.shouldRetrySummary(text, firstResult)) {
                    engine.runLlmSummaryRetry(
                        text = text,
                        previousAnswer = firstResult.retryContext(),
                        maxInputTokens = maxInputTokens,
                        maxNewTokens = maxNewTokens,
                    )
                } else {
                    firstResult
                }
            processor.normalizeSummary(llmResult, sourceText = text)
        }

    override suspend fun suggestTags(
        text: String,
        maxInputTokens: Int,
        maxTags: Int,
    ): Set<String> =
        withContext(Dispatchers.Default) {
            val firstResult =
                engine.runLlmTagging(
                    text = text,
                    maxInputTokens = maxInputTokens,
                    maxTags = maxTags,
                )
            val llmResult =
                if (processor.shouldRetryTags(text, firstResult, maxTags)) {
                    engine.runLlmTaggingRetry(
                        text = text,
                        previousAnswer = firstResult.retryContext(),
                        maxInputTokens = maxInputTokens,
                        maxTags = maxTags,
                    )
                } else {
                    firstResult
                }
            processor.normalizeTags(llmResult, maxTags, sourceText = text)
        }

    override suspend fun rewrite(
        text: String,
        style: RewriteStyle,
        maxInputTokens: Int,
        maxNewTokens: Int,
    ): String =
        withContext(Dispatchers.Default) {
            val firstResult =
                engine.runLlmRewrite(
                    text = text,
                    style = style,
                    maxInputTokens = maxInputTokens,
                    maxNewTokens = maxNewTokens,
                )
            val llmResult =
                if (processor.shouldRetryRewrite(text, firstResult)) {
                    engine.runLlmRewriteRetry(
                        text = text,
                        style = style,
                        previousAnswer = firstResult.retryContext(),
                        maxInputTokens = maxInputTokens,
                        maxNewTokens = maxNewTokens,
                    )
                } else {
                    firstResult
                }
            processor.normalizeRewrite(llmResult, sourceText = text)
        }

    override suspend fun tagIMGs(img: List<String>): Set<String> = imageTaggingBackend.tagImages(img)

    private fun String.retryContext(): String = trim().take(MAX_RETRY_CONTEXT_CHARS)

    private companion object {
        const val MAX_RETRY_CONTEXT_CHARS = 600
    }
}

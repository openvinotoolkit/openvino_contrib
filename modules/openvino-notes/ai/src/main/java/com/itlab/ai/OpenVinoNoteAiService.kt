package com.itlab.ai

import com.itlab.domain.ai.NoteAiService

class OpenVinoNoteAiService(
    private val engine: OpenVinoEngine,
    private val processor: ResultProcessor,
) : NoteAiService {
    override suspend fun summarize(text: String): String {
        val llmResult = engine.runLlmSummary(text)
        return processor.normalizeSummary(llmResult)
    }

    override suspend fun tagTXT(text: String): Set<String> {
        val llmResult = engine.runLlmTagging(text)
        return processor.normalizeTags(llmResult)
    }

    override suspend fun tagIMGs(img: List<String>): Set<String> =
        img
            .map { source -> engine.runYoloTagging(source) }
            .flatMap { result -> processor.normalizeTags(result) }
            .toSet()
}

package com.itlab.ai

import com.itlab.domain.ai.RewriteStyle
import kotlin.math.max

class NoteLlmPromptBuilder(
    private val config: OnDeviceLlmConfig = OnDeviceLlmConfig.defaultAndroid(),
) {
    fun summaryPrompt(
        text: String,
        maxInputTokens: Int = config.summaryMaxInputTokens,
    ): String {
        val note = trimInput(text, maxInputTokens)
        val language = NoteLanguageDetector.detect(note)
        return chatPrompt(
            """
            ${language.summaryInstruction}
            Detected note language: ${language.displayName}.
            Answer only in ${language.answerInstruction}.
            Start immediately with the final summary.
            Use at most 25 words and end with a complete sentence.
            Write a useful summary, not a copy of the first line or the full note.
            Capture the main action or decision plus the most important owner, deadline, place, risk, or condition when present.
            Use an extractive style: keep source wording for key facts and remove less important details.
            Avoid synonyms and paraphrases when a source phrase is already natural.
            If grammar is uncertain, use an exact complete source sentence as the summary.
            Do not invent facts and do not output tags.
            Do not add generic statements, causes, or interpretations that are not written in the note.
            Never stop after an abbreviation such as Dr., Mr., or Prof.; include the following name.
            Do not translate names, locations, dates, times, numbers, or product names.
            Include the most important who/what/when/where fact if it is present.
            Return only the summary, without markdown, analysis, acknowledgements, or a preamble.

            Note:
            $note

            ${language.summaryCue}:
            """.trimIndent(),
        )
    }

    @Suppress("UNUSED_PARAMETER")
    fun summaryRetryPrompt(
        text: String,
        previousAnswer: String,
        maxInputTokens: Int = config.summaryMaxInputTokens,
    ): String {
        val note = trimInput(text, maxInputTokens)
        val language = NoteLanguageDetector.detect(note)
        return chatPrompt(
            """
            The previous summary was invalid because it was empty, too long, repeated, or used the wrong language.
            Rewrite it once more and do not copy the invalid previous answer.
            ${language.summaryInstruction}
            Answer only in ${language.answerInstruction}.
            Use at most 20 words.
            Summarize the main action or decision plus the most important concrete fact.
            Use an extractive style: keep source wording for key facts and remove less important details.
            Avoid synonyms and paraphrases when a source phrase is already natural.
            If grammar is uncertain, use an exact complete source sentence as the summary.
            Keep names, locations, dates, times, numbers, and product names unchanged.
            Do not invent facts and do not output tags.
            Do not add generic statements, causes, or interpretations that are not written in the note.
            Never stop after an abbreviation such as Dr., Mr., or Prof.; include the following name.
            Return only the corrected summary. Do not explain the correction.

            Note:
            $note

            ${language.summaryCue}:
            """.trimIndent(),
        )
    }

    fun tagsPrompt(
        text: String,
        maxInputTokens: Int = config.tagsMaxInputTokens,
        maxTags: Int = config.maxTags,
    ): String {
        val note = trimInput(text, maxInputTokens)
        val language = NoteLanguageDetector.detect(note)
        return chatPrompt(
            """
            ${language.tagsInstruction}
            Suggest up to $maxTags complete topic tags for the note.
            Detected note language: ${language.displayName}.
            Prefer tags in ${language.answerInstruction}.
            Use complete words copied from the note when possible.
            Preserve proper nouns and product names exactly when they are useful tags.
            Do not abbreviate. Do not output single letters or initials.
            Each tag must be at most two words and 32 characters.
            ${language.tagExample}
            Return only a comma-separated tag list. Use lowercase tags except proper nouns.
            Do not add explanations, analysis, acknowledgements, or a preamble.

            Note:
            $note

            ${language.tagsCue}:
            """.trimIndent(),
        )
    }

    fun tagsRetryPrompt(
        text: String,
        previousAnswer: String,
        maxInputTokens: Int = config.tagsMaxInputTokens,
        maxTags: Int = config.maxTags,
    ): String {
        val note = trimInput(text, maxInputTokens)
        val language = NoteLanguageDetector.detect(note)
        return chatPrompt(
            """
            The previous tag list was invalid because it contained sentences, ungrounded words, or the wrong language.
            Rewrite the tags once more.
            ${language.tagsInstruction}
            Suggest up to $maxTags tags.
            Use only words that appear in the note when possible.
            Each tag must be at most two words and 32 characters.
            Return only comma-separated tags. Do not explain the correction.
            ${language.tagExample}

            Invalid previous answer:
            $previousAnswer

            Note:
            $note

            ${language.tagsCue}:
            """.trimIndent(),
        )
    }

    fun rewritePrompt(
        text: String,
        style: RewriteStyle,
        maxInputTokens: Int = config.rewriteMaxInputTokens,
    ): String {
        val note = trimInput(text, maxInputTokens)
        val language = NoteLanguageDetector.detect(note)
        return chatPrompt(
            """
            ${rewriteInstruction(style, language)}
            Detected note language: ${language.displayName}.
            Answer only in ${language.answerInstruction}.
            Do not summarize or translate the note.
            Make the smallest possible edit that improves readability.
            Prefer preserving the original sentence structure when it is already clear.
            It is valid to return the original note unchanged if it is already clear and grammatical.
            Improve grammar, punctuation, sentence boundaries, and ordering while preserving the same level of detail.
            Keep original wording for correct phrases; only fix grammar, punctuation, and ordering where needed.
            Do not replace a correct noun phrase with a new verb phrase.
            Do not shorten the note into a summary.
            Preserve every named person, location, deadline, number, condition, and user intent.
            Keep all actionable items, constraints, risks, and follow-up steps.
            Keep concrete words from the note when they carry facts, names, dates, places, or product names.
            If the note is already short, clean it up without changing its facts.
            Prefer complete sentences unless the original note is a checklist.
            Output only the rewritten note text.
            Do not copy any instruction, label, markdown, analysis, acknowledgement, or preamble from this prompt.

            Note:
            $note

            ${language.rewriteCue}:
            """.trimIndent(),
        )
    }

    @Suppress("UNUSED_PARAMETER")
    fun rewriteRetryPrompt(
        text: String,
        style: RewriteStyle,
        previousAnswer: String,
        maxInputTokens: Int = config.rewriteMaxInputTokens,
    ): String {
        val note = trimInput(text, maxInputTokens)
        val language = NoteLanguageDetector.detect(note)
        return chatPrompt(
            """
            The previous rewrite was invalid because it used the wrong language, lost facts, or included artifacts.
            Rewrite the original note again, ignoring the invalid previous answer.
            ${rewriteInstruction(style, language)}
            Answer only in ${language.answerInstruction}.
            Make the smallest possible edit that improves readability.
            Prefer preserving the original sentence structure when it is already clear.
            It is valid to return the original note unchanged if it is already clear and grammatical.
            Improve grammar, punctuation, sentence boundaries, and ordering while preserving the same level of detail.
            Keep original wording for correct phrases; only fix grammar, punctuation, and ordering where needed.
            Do not replace a correct noun phrase with a new verb phrase.
            Do not shorten the note into a summary.
            Preserve every named person, location, deadline, number, condition, and user intent.
            Keep all actionable items, constraints, risks, and follow-up steps.
            Keep concrete words from the note when they carry facts, names, dates, places, or product names.
            Prefer complete sentences unless the original note is a checklist.
            Output only the corrected note text.
            Do not copy any instruction, label, markdown, analysis, acknowledgement, or preamble from this prompt.

            Note:
            $note

            ${language.rewriteCue}:
            """.trimIndent(),
        )
    }

    private fun chatPrompt(instruction: String): String = instruction

    private fun rewriteInstruction(
        style: RewriteStyle,
        language: NoteLanguage,
    ): String =
        when (style) {
            RewriteStyle.CLEANUP -> language.rewriteInstruction
        }

    private fun trimInput(
        text: String,
        maxInputTokens: Int,
    ): String =
        text
            .trim()
            .take(minOf(config.maxInputChars, max(1, maxInputTokens) * config.approximateCharsPerToken))
}

package com.itlab.ai

class ResultProcessor {
    fun normalizeSummary(
        raw: String,
        sourceText: String? = null,
    ): String {
        val normalized =
            GeneratedTextCleaner
                .normalizeTextAnswer(raw)
                .let { answer ->
                    if (sourceText == null) {
                        answer
                    } else {
                        val compacted = AnswerQuality.compactGeneratedSummary(sourceText, answer)
                        GeneratedTextCleaner.repairLanguageSpecificAnswer(sourceText, compacted)
                    }
                }
        return if (
            sourceText != null &&
            (
                AnswerQuality.shouldPreferExtractiveSummary(sourceText, normalized) ||
                    !AnswerQuality.isAcceptableSummary(sourceText, normalized)
            )
        ) {
            AnswerQuality.extractiveSummary(sourceText)
        } else {
            normalized
        }
    }

    fun normalizeRewrite(
        raw: String,
        sourceText: String? = null,
    ): String {
        val normalized =
            GeneratedTextCleaner
                .normalizeTextAnswer(raw)
                .let { answer ->
                    if (sourceText == null) {
                        answer
                    } else {
                        GeneratedTextCleaner.repairLanguageSpecificAnswer(sourceText, answer)
                    }
                }
        return if (sourceText != null && !AnswerQuality.isAcceptableRewrite(sourceText, normalized)) {
            AnswerQuality.cleanRewriteFallback(sourceText)
        } else {
            normalized
        }
    }

    fun normalizeTags(
        raw: String,
        maxTags: Int = Int.MAX_VALUE,
        sourceText: String? = null,
    ): Set<String> = TagProcessor.normalizeTags(raw, maxTags, sourceText)

    fun shouldRetrySummary(
        sourceText: String,
        raw: String,
    ): Boolean = !AnswerQuality.isAcceptableSummary(sourceText, GeneratedTextCleaner.normalizeTextAnswer(raw))

    fun shouldRetryRewrite(
        sourceText: String,
        raw: String,
    ): Boolean = !AnswerQuality.isAcceptableRewrite(sourceText, GeneratedTextCleaner.normalizeTextAnswer(raw))

    fun shouldRetryTags(
        sourceText: String,
        raw: String,
        maxTags: Int,
    ): Boolean = TagProcessor.shouldRetryTags(sourceText, raw, maxTags)
}

private object AnswerQuality {
    fun isAcceptableSummary(
        sourceText: String,
        answer: String,
    ): Boolean =
        answer.isNotBlank() &&
            answer.length <= MAX_GENERATED_SUMMARY_CHARS &&
            hasCompleteTextEnding(answer) &&
            !containsUnsupportedGenericClaim(sourceText, answer) &&
            !shouldUseSourceFallback(sourceText, answer)

    fun isAcceptableRewrite(
        sourceText: String,
        answer: String,
    ): Boolean =
        answer.isNotBlank() &&
            hasCompleteTextEnding(answer) &&
            !containsUnsupportedGenericClaim(sourceText, answer) &&
            !shouldUseSourceFallback(sourceText, answer) &&
            !containsPromptArtifact(sourceText, answer) &&
            preservesRewriteAnchors(sourceText, answer)

    fun extractiveSummary(sourceText: String): String {
        val sentences = sourceText.extractSummarySentences()
        if (sentences.isEmpty()) {
            return ""
        }

        val language = NoteLanguageDetector.detect(sourceText)
        val ranked =
            sentences
                .mapIndexed { index, sentence ->
                    SummarySentence(
                        text = sentence,
                        index = index,
                        score = summarySentenceScore(sentence, index, language),
                    )
                }.sortedWith(
                    compareByDescending<SummarySentence> { it.score }
                        .thenBy { it.index },
                )
        val best = ranked.first()
        val selected =
            ranked
                .filter { candidate ->
                    candidate.index == best.index ||
                        (
                            candidate.score >= MIN_SECONDARY_SUMMARY_SCORE &&
                                candidate.score >= best.score - SECONDARY_SUMMARY_SCORE_GAP
                        )
                }.take(MAX_EXTRACTIVE_SUMMARY_SENTENCES)
                .sortedBy { it.index }

        return selected
            .joinToString(" ") { it.text }
            .trim()
            .take(MAX_EXTRACTIVE_SUMMARY_CHARS)
            .trimEnd(',', ';', ':', '-')
    }

    fun compactGeneratedSummary(
        sourceText: String,
        answer: String,
    ): String {
        val sentences = answer.extractSummarySentences()
        if (sentences.size <= 1) {
            return answer
        }

        val maxChars =
            minOf(
                MAX_GENERATED_SUMMARY_CHARS,
                (sourceText.length * 2 / 3).coerceAtLeast(MIN_COMPACT_GENERATED_SUMMARY_CHARS),
            )
        val selected = mutableListOf<String>()
        for (sentence in sentences) {
            val candidate = (selected + sentence).joinToString(" ")
            if (candidate.length <= maxChars || selected.isEmpty()) {
                selected += sentence
            }
            if (selected.size >= MAX_GENERATED_SUMMARY_SENTENCES) {
                break
            }
        }

        return selected.joinToString(" ").trim()
    }

    fun shouldPreferExtractiveSummary(
        sourceText: String,
        answer: String,
    ): Boolean {
        if (NoteLanguageDetector.detect(sourceText) != NoteLanguage.RUSSIAN) {
            return false
        }
        val normalizedSource = GeneratedTextCleaner.normalizeForMatching(sourceText)
        val normalizedAnswer = GeneratedTextCleaner.normalizeForMatching(answer)
        return normalizedAnswer.isNotBlank() && !normalizedSource.contains(normalizedAnswer)
    }

    fun cleanRewriteFallback(sourceText: String): String =
        sourceText
            .trim()
            .lineSequence()
            .map(::cleanRewriteFallbackLine)
            .filter { it.isNotBlank() }
            .joinToString("\n")

    private fun shouldUseSourceFallback(
        sourceText: String,
        answer: String,
    ): Boolean {
        if (answer.isBlank()) {
            return true
        }
        val language = NoteLanguageDetector.detect(sourceText)
        return language.forbiddenOutputPattern?.containsMatchIn(answer) == true
    }

    private fun hasCompleteTextEnding(answer: String): Boolean {
        val trimmed = answer.trim()
        return trimmed.lastOrNull() in terminalPunctuation &&
            !trailingIncompleteAbbreviation.containsMatchIn(trimmed)
    }

    private fun containsPromptArtifact(
        sourceText: String,
        answer: String,
    ): Boolean =
        rewritePromptArtifacts.any { artifact ->
            artifact.containsMatchIn(answer) && !artifact.containsMatchIn(sourceText)
        }

    private fun containsUnsupportedGenericClaim(
        sourceText: String,
        answer: String,
    ): Boolean =
        genericClaimArtifacts.any { artifact ->
            artifact.containsMatchIn(answer) && !artifact.containsMatchIn(sourceText)
        }

    private fun preservesRewriteAnchors(
        sourceText: String,
        answer: String,
    ): Boolean {
        val anchors = extractRewriteAnchors(sourceText)
        if (anchors.isEmpty()) {
            return true
        }

        val normalizedAnswer = GeneratedTextCleaner.normalizeForMatching(answer)
        val requiredAnchors = anchors.filter { it.required }
        if (requiredAnchors.isEmpty() && anchors.size < MIN_ANCHORS_FOR_OPTIONAL_MATCHING) {
            return true
        }
        if (requiredAnchors.any { !it.matches(normalizedAnswer) }) {
            return false
        }

        val matchedAnchors = anchors.count { it.matches(normalizedAnswer) }
        val minimumMatches =
            when {
                anchors.size <= 2 -> 1
                anchors.size <= 5 -> 2
                else -> anchors.size / 2
            }
        return matchedAnchors >= minimumMatches
    }

    private fun String.extractSummarySentences(): List<String> {
        val trimmed = trim()
        if (trimmed.isBlank()) {
            return emptyList()
        }
        return trimmed
            .split(sentenceBoundary)
            .map(::cleanSummarySentence)
            .filter { it.isNotBlank() }
            .ifEmpty { listOf(cleanSummarySentence(trimmed)) }
    }

    private fun cleanSummarySentence(sentence: String): String =
        sentence
            .trim()
            .replace(horizontalWhitespace, " ")
            .trim(',', ';', ':', '-')

    private fun cleanRewriteFallbackLine(line: String): String {
        val compact =
            line
                .trim()
                .replace(horizontalWhitespace, " ")
                .replace(spaceBeforePunctuation, "$1")
                .replace(spaceAfterCommaLikePunctuation, "$1 ")
                .replace(spaceAfterPeriodBeforeLetter, ". ")
                .replace(spaceAfterNonTimeColon, ": ")
                .trim()

        return if (compact.lastOrNull()?.isLetterOrDigit() == true) {
            "$compact."
        } else {
            compact
        }
    }

    private fun summarySentenceScore(
        sentence: String,
        index: Int,
        language: NoteLanguage,
    ): Int {
        val normalized = GeneratedTextCleaner.normalizeForMatching(sentence)
        var score = 0

        if (sentence.length in GOOD_SUMMARY_SENTENCE_LENGTH_RANGE) {
            score += 4
        }
        if (sentence.any { it.isDigit() }) {
            score += 8
        }
        if (timeOrDateCue.containsMatchIn(sentence)) {
            score += 8
        }
        if (conditionalCue.containsMatchIn(sentence)) {
            score += 5
        }
        if (actionCue.containsMatchIn(sentence)) {
            score += 6
        }
        if (hasNamedOrProductAnchor(sentence)) {
            score += 8
        }

        score +=
            summarySalientFragments
                .count { normalized.contains(it) }
                .coerceAtMost(MAX_SUMMARY_SALIENT_MATCHES) * SUMMARY_SALIENT_MATCH_SCORE

        if (index == 0) {
            score += 1
        }
        if (sentence.length < MIN_USEFUL_SUMMARY_SENTENCE_CHARS) {
            score -= 8
        }
        if (genericSummaryLead.containsMatchIn(sentence)) {
            score -= 12
        }
        if (normalized in language.stopWords) {
            score -= 6
        }

        return score
    }

    private fun hasNamedOrProductAnchor(sentence: String): Boolean =
        sourceWordRegex
            .findAll(sentence)
            .drop(1)
            .any { match ->
                val raw = match.value
                raw.firstOrNull()?.isUpperCase() == true ||
                    (
                        raw.length > 1 &&
                            raw.any { it.isUpperCase() } &&
                            raw.drop(1).any { it.isUpperCase() || it.isDigit() }
                    )
            }

    private fun extractRewriteAnchors(sourceText: String): List<RewriteAnchor> {
        val language = NoteLanguageDetector.detect(sourceText)
        return sourceWordRegex
            .findAll(sourceText)
            .mapIndexedNotNull { index, match ->
                val raw = match.value.trim('.', ',', ';', ':', '!', '?', '(', ')', '[', ']', '"', '\'')
                val normalized = GeneratedTextCleaner.normalizeForMatching(raw)
                if (!isUsefulRewriteAnchor(normalized, language)) {
                    null
                } else {
                    RewriteAnchor(
                        normalized = normalized,
                        required = isRequiredRewriteAnchor(index, raw, normalized),
                    )
                }
            }.distinctBy { it.normalized }
            .take(MAX_REWRITE_ANCHORS)
            .toList()
    }

    private fun isUsefulRewriteAnchor(
        normalized: String,
        language: NoteLanguage,
    ): Boolean =
        normalized.length >= MIN_REWRITE_ANCHOR_LENGTH &&
            normalized !in commonRewriteStopWords &&
            normalized !in language.stopWords

    private fun isRequiredRewriteAnchor(
        index: Int,
        raw: String,
        normalized: String,
    ): Boolean {
        val hasDigit = raw.any { it.isDigit() }
        val hasProductShape =
            raw.length > 1 &&
                raw.any { it.isUpperCase() } &&
                raw.drop(1).any { it.isUpperCase() || it.isDigit() }
        val hasProperNounShape =
            index > 0 &&
                raw.firstOrNull()?.isUpperCase() == true &&
                raw.drop(1).any { it.isLowerCase() }
        return hasDigit ||
            hasProductShape ||
            hasProperNounShape ||
            requiredRewriteFragments.any { normalized.contains(it) }
    }

    private const val MAX_GENERATED_SUMMARY_CHARS = 260
    private const val MAX_GENERATED_SUMMARY_SENTENCES = 2
    private const val MIN_COMPACT_GENERATED_SUMMARY_CHARS = 120
    private const val MAX_EXTRACTIVE_SUMMARY_CHARS = 240
    private const val MAX_EXTRACTIVE_SUMMARY_SENTENCES = 2
    private const val MIN_SECONDARY_SUMMARY_SCORE = 12
    private const val SECONDARY_SUMMARY_SCORE_GAP = 6
    private const val MIN_USEFUL_SUMMARY_SENTENCE_CHARS = 20
    private const val MAX_SUMMARY_SALIENT_MATCHES = 3
    private const val SUMMARY_SALIENT_MATCH_SCORE = 5
    private const val MIN_REWRITE_ANCHOR_LENGTH = 4
    private const val MIN_ANCHORS_FOR_OPTIONAL_MATCHING = 3
    private const val MAX_REWRITE_ANCHORS = 12
    private val GOOD_SUMMARY_SENTENCE_LENGTH_RANGE = 40..220
    private val sentenceBoundary =
        Regex(
            """(?<!dr\.)(?<!mr\.)(?<!ms\.)(?<!mrs\.)(?<!prof\.)(?<=[.!?。！？])\s+""",
            RegexOption.IGNORE_CASE,
        )
    private val horizontalWhitespace = Regex("""[ \t\x0B\f\r]+""")
    private val spaceBeforePunctuation = Regex("""\s+([,.;:!?])""")
    private val spaceAfterCommaLikePunctuation = Regex("""([,;!?])(?=\S)""")
    private val spaceAfterPeriodBeforeLetter = Regex("""\.(?=\p{L})""")
    private val spaceAfterNonTimeColon = Regex("""(?<!\d):(?!\d)(?=\S)""")
    private val timeOrDateCue =
        Regex(
            """\d{1,2}:\d{2}|\d{1,2}[./-]\d{1,2}|""" +
                """\b(?:monday|tuesday|wednesday|thursday|friday|samstag|sonntag|""" +
                """montag|dienstag|mittwoch|donnerstag|freitag|lundi|mardi|mercredi|""" +
                """jeudi|vendredi|понедельник|вторник|среду|среда|четверг|пятницу|пятница)\b""",
            RegexOption.IGNORE_CASE,
        )
    private val conditionalCue = Regex("""\b(?:if|when|unless|wenn|falls|si|если|когда)\b""", RegexOption.IGNORE_CASE)
    private val actionCue =
        Regex(
            """\b(?:must|should|needs?|confirm|prepare|send|review|check|test|""" +
                """muss|soll|prüft|prueft|testet|koordiniert|prépare|prepare|vérifie|verifie|""" +
                """нужно|надо|проверяет|проверить|готовит|отправить|согласовать)\b""",
            RegexOption.IGNORE_CASE,
        )
    private val genericSummaryLead =
        Regex(
            """^\s*(?:notes?|meeting notes|draft|todo|заметк[аи]|черновик|""" +
                """notiz|note|résumé)\b[:：.]?""",
            RegexOption.IGNORE_CASE,
        )
    private val terminalPunctuation = setOf('.', '!', '?', '。', '！', '？')
    private val trailingIncompleteAbbreviation =
        Regex("""\b(?:dr|mr|mrs|ms|prof)\.$""", RegexOption.IGNORE_CASE)
    private val sourceWordRegex = Regex("""[\p{L}\p{N}][\p{L}\p{N}_:+.-]{1,48}""")
    private val summarySalientFragments =
        listOf(
            "openvino",
            "android",
            "qwen",
            "firebase",
            "gradle",
            "release",
            "debug",
            "model",
            "risk",
            "sensor",
            "robot",
            "prototype",
            "pressure",
            "battery",
            "security",
            "модель",
            "стенд",
            "датчик",
            "риск",
            "запуск",
            "склад",
            "risiko",
            "risik",
            "akku",
            "qualität",
            "qualitaet",
            "risques",
            "pression",
            "sécurité",
        )
    private val rewritePromptArtifacts =
        listOf(
            Regex("""\bstart immediately with the rewritten note\b""", RegexOption.IGNORE_CASE),
            Regex("""\breturn only\b""", RegexOption.IGNORE_CASE),
            Regex("""\banswer only in\b""", RegexOption.IGNORE_CASE),
            Regex("""\binvalid previous answer\b""", RegexOption.IGNORE_CASE),
            Regex("""\bprevious rewrite was invalid\b""", RegexOption.IGNORE_CASE),
            Regex("""\bdo not copy any instruction\b""", RegexOption.IGNORE_CASE),
            Regex("""\bdo not explain\b""", RegexOption.IGNORE_CASE),
            Regex("""^\s*(?:task|note|user note)\s*:""", RegexOption.IGNORE_CASE),
        )
    private val genericClaimArtifacts =
        listOf(
            Regex("""\bin der regel\b""", RegexOption.IGNORE_CASE),
            Regex("""\b(?:usually|typically|generally|as a rule)\b""", RegexOption.IGNORE_CASE),
            Regex("""\b(?:обычно|как правило|в целом)\b""", RegexOption.IGNORE_CASE),
            Regex("""\b(?:en général|généralement)\b""", RegexOption.IGNORE_CASE),
        )
    private val commonRewriteStopWords =
        setOf(
            "this",
            "that",
            "with",
            "from",
            "have",
            "will",
            "надо",
            "нужно",
            "если",
            "еще",
            "ещё",
            "eine",
            "einen",
            "soll",
            "muss",
            "avec",
            "pour",
            "avant",
            "doit",
        )
    private val requiredRewriteFragments =
        listOf(
            "openvino",
            "android",
            "qwen",
            "firebase",
            "gradle",
            "release",
            "debug",
            "модель",
            "опенвино",
        )

    private data class RewriteAnchor(
        val normalized: String,
        val required: Boolean,
    ) {
        fun matches(normalizedAnswer: String): Boolean =
            normalizedAnswer.contains(normalized) ||
                (normalized.length >= 6 && normalizedAnswer.contains(normalized.take(6)))
    }

    private data class SummarySentence(
        val text: String,
        val index: Int,
        val score: Int,
    )
}

private object GeneratedTextCleaner {
    fun normalizeTextAnswer(raw: String): String =
        stripLeadingInstructionEcho(stripLeadingAssistantLabel(stripGeneratedMetadataSections(raw)))
            .stripWrappingMarkdown()
            .trim()

    fun repairLanguageSpecificAnswer(
        sourceText: String,
        answer: String,
    ): String =
        when (NoteLanguageDetector.detect(sourceText)) {
            NoteLanguage.RUSSIAN -> repairRussianCases(answer)
            NoteLanguage.GERMAN -> repairGermanDiacritics(answer)
            NoteLanguage.ENGLISH,
            NoteLanguage.FRENCH,
            -> answer
        }

    fun stripLeadingAssistantLabel(raw: String): String =
        raw
            .replace(leadingAssistantLabel, "")
            .replace(leadingPunctuationOnlyLine, "")
            .trim()

    private fun stripLeadingInstructionEcho(raw: String): String =
        raw
            .replace(leadingInstructionEcho, "")
            .trim()

    fun normalizeForMatching(text: String): String =
        text
            .lowercase()
            .replace('ё', 'е')
            .replace(diacriticInsensitivePunctuation, "")

    private fun repairRussianCases(answer: String): String =
        russianCaseRepairs.fold(answer) { current, (pattern, replacement) ->
            current.replace(pattern, replacement)
        }

    private fun repairGermanDiacritics(answer: String): String =
        germanDiacriticRepairs.fold(answer) { current, (pattern, replacement) ->
            current.replace(pattern, replacement)
        }

    private fun stripGeneratedMetadataSections(raw: String): String =
        raw
            .lineSequence()
            .withIndex()
            .takeWhile { (index, line) -> index == 0 || !generatedMetadataHeader.matches(line) }
            .map { it.value }
            .joinToString("\n")
            .trim()

    private fun String.stripWrappingMarkdown(): String {
        val value = trim()
        return wrappingMarkdown
            .matchEntire(value)
            ?.groupValues
            ?.get(1)
            ?.trim()
            ?: value
    }

    private val generatedMetadataHeader =
        Regex(
            pattern = """\s*(?:[-*]\s*)?(?:\*\*)?\s*(?:tags?|summary|title)\s*(?:\*\*)?\s*:.*""",
            option = RegexOption.IGNORE_CASE,
        )
    private val leadingAssistantLabel =
        Regex(
            pattern =
                """^\s*(?:[-*]\s*)?(?:\*\*)?\s*""" +
                    """(?:(?:english|deutsche?)\s+)?""" +
                    """(?:summary|tags?|rewrite|rewritten note|резюме|теги|переписанный текст|""" +
                    """zusammenfassung|stichworte|ueberarbeitete notiz|überarbeitete notiz|""" +
                    """mots-cl[eé]s|r[eé]sum[eé]|synth[eè]se|note r[eé][eé]crite)""" +
                    """\s*(?:\*\*)?\s*:\s*""",
            option = RegexOption.IGNORE_CASE,
        )
    private val leadingPunctuationOnlyLine = Regex("""^\s*[:：]\s*(?:\R+|$)""")
    private val leadingInstructionEcho =
        Regex(
            pattern =
                """^\s*(?:[-*]\s*)?(?:start immediately with the rewritten note|""" +
                    """return only the rewritten note text|output only the rewritten note text)\s*[.!:]?\s*(?:\R+|$)""",
            option = RegexOption.IGNORE_CASE,
        )
    private val wrappingMarkdown = Regex("""(?:\*\*|__)(.*)(?:\*\*|__)""", RegexOption.DOT_MATCHES_ALL)
    private val diacriticInsensitivePunctuation = Regex("""[\p{Punct}\s]+""")
    private val russianCaseRepairs =
        listOf(
            Regex("""\b(проверяет|запускает|тестирует|готовит)\s+пилотного стенда\b""") to
                "$1 пилотный стенд",
            Regex("""\b(проверить|запустить|тестировать|подготовить)\s+пилотного стенда\b""") to
                "$1 пилотный стенд",
        )
    private val germanDiacriticRepairs =
        listOf(
            Regex("""\bFur\b""") to "Für",
            Regex("""\bfuer\b""", RegexOption.IGNORE_CASE) to "für",
            Regex("""\bUeber""") to "Über",
            Regex("""\bueber""") to "über",
            Regex("""Qualitaet""") to "Qualität",
            Regex("""Qualitaets""") to "Qualitäts",
            Regex("""Pruefung""") to "Prüfung",
            Regex("""pruefung""") to "prüfung",
        )
}

private object TagProcessor {
    fun normalizeTags(
        raw: String,
        maxTags: Int,
        sourceText: String?,
    ): Set<String> {
        val generatedTags = generatedTags(raw, sourceText)
        if (sourceText == null) {
            return generatedTags.take(maxTags).toCollection(LinkedHashSet())
        }

        return (generatedTags + extractKeywordTags(sourceText, maxTags * 2))
            .distinct()
            .take(maxTags)
            .toCollection(LinkedHashSet())
    }

    fun shouldRetryTags(
        sourceText: String,
        raw: String,
        maxTags: Int,
    ): Boolean =
        generatedTags(raw, sourceText).take(maxTags).size <
            minOf(MIN_GENERATED_TAGS_BEFORE_FALLBACK, maxTags)

    private fun generatedTags(
        raw: String,
        sourceText: String?,
    ): List<String> {
        val sourceForMatching = sourceText?.let(GeneratedTextCleaner::normalizeForMatching)
        return GeneratedTextCleaner
            .stripLeadingAssistantLabel(raw)
            .split(',', '\n', ';', '，')
            .map(::cleanTag)
            .filter(::isUsefulTag)
            .filter { tag ->
                sourceForMatching == null || isGroundedIn(tag, sourceForMatching)
            }
    }

    private fun extractKeywordTags(
        text: String,
        maxTags: Int,
    ): Set<String> {
        if (text.isBlank() || maxTags <= 0) {
            return emptySet()
        }
        val language = NoteLanguageDetector.detect(text)
        return wordRegex
            .findAll(text)
            .mapIndexed { index, match ->
                val rawToken = match.value.trim('-', '_', '.', ',')
                val tag = cleanTag(rawToken)
                TagCandidate(
                    tag = tag,
                    index = index,
                    score = keywordScore(tag, rawToken, language),
                )
            }.filter { isUsefulTag(it.tag) }
            .filterNot { it.tag in commonStopWords || it.tag in language.stopWords }
            .distinctBy { it.tag }
            .sortedWith(
                compareByDescending<TagCandidate> { it.score }
                    .thenBy { it.index },
            ).map { it.tag }
            .take(maxTags)
            .toCollection(LinkedHashSet())
    }

    private fun cleanTag(raw: String): String =
        raw
            .trim()
            .replace(leadingTagDecoration, "")
            .trim()
            .trim('"', '\'', '`', '.', ':', '-')
            .lowercase()

    private fun isUsefulTag(value: String): Boolean {
        val wordCount = value.split(whitespace).count { it.isNotBlank() }
        return value.length in MIN_TAG_LENGTH..MAX_TAG_LENGTH &&
            !value.contains("<|") &&
            !value.contains("*") &&
            !sentencePunctuation.containsMatchIn(value) &&
            wordCount <= MAX_TAG_WORDS &&
            value.count { it.isLetterOrDigit() } >= MIN_TAG_LETTER_OR_DIGIT_COUNT
    }

    private fun isGroundedIn(
        tag: String,
        normalizedSource: String,
    ): Boolean =
        tag
            .split(whitespace)
            .filter { it.isNotBlank() }
            .all { word ->
                val normalizedWord = GeneratedTextCleaner.normalizeForMatching(word)
                normalizedWord.length <= 2 ||
                    normalizedSource.contains(normalizedWord) ||
                    normalizedSource.contains(normalizedWord.take(minOf(5, normalizedWord.length)))
            }

    private fun keywordScore(
        tag: String,
        rawToken: String,
        language: NoteLanguage,
    ): Int {
        var score = 0
        if (salientFragments.any { tag.contains(it, ignoreCase = true) }) {
            score += 20
        }
        if (rawToken.any { it.isUpperCase() } || rawToken.any { it.isDigit() }) {
            score += 4
        }
        if (tag.length in 5..18) {
            score += 2
        }
        if (language == NoteLanguage.RUSSIAN && tag.any { it in 'а'..'я' || it == 'ё' }) {
            score += 2
        }
        return score
    }

    private data class TagCandidate(
        val tag: String,
        val index: Int,
        val score: Int,
    )

    private const val MIN_TAG_LENGTH = 2
    private const val MAX_TAG_LENGTH = 32
    private const val MAX_TAG_WORDS = 2
    private const val MIN_TAG_LETTER_OR_DIGIT_COUNT = 2
    private const val MIN_GENERATED_TAGS_BEFORE_FALLBACK = 2
    private val leadingTagDecoration = Regex("""^\s*(?:[-*#]|\d+[.)])\s*""")
    private val whitespace = Regex("""\s+""")
    private val sentencePunctuation = Regex("""[.!?]""")
    private val wordRegex = Regex("""[\p{L}\p{N}][\p{L}\p{N}_-]{1,31}""")
    private val commonStopWords =
        setOf(
            "and",
            "the",
            "for",
            "with",
            "если",
            "для",
            "und",
            "der",
            "die",
            "das",
            "les",
            "des",
            "une",
        )
    private val salientFragments =
        listOf(
            "openvino",
            "model",
            "модель",
            "датчик",
            "sensor",
            "risk",
            "risik",
            "риск",
            "demo",
            "robot",
            "prototype",
            "pressure",
            "pression",
            "temperatur",
            "qualität",
            "quality",
            "sécurité",
            "склад",
            "стенд",
            "запуск",
            "battery",
            "акку",
            "akku",
        )
}

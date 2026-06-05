package com.itlab.ai

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class ResultProcessorTest {
    @Test
    fun normalizeSummary_trimsSpaces() {
        val processor = ResultProcessor()

        val result = processor.normalizeSummary("  short summary  ")

        assertEquals("short summary", result)
    }

    @Test
    fun normalizeSummary_removesGeneratedMetadataSections() {
        val processor = ResultProcessor()

        val result =
            processor.normalizeSummary(
                """
                Short summary.
                **Tags**: android, openvino
                """.trimIndent(),
            )

        assertEquals("Short summary.", result)
    }

    @Test
    fun normalizeSummary_removesAssistantLabel() {
        val processor = ResultProcessor()

        val result = processor.normalizeSummary("Summary: Марина проверяет стенд в Казани.")

        assertEquals("Марина проверяет стенд в Казани.", result)
    }

    @Test
    fun normalizeSummary_removesWrappingMarkdown() {
        val processor = ResultProcessor()

        val result = processor.normalizeSummary("**Frau Müller prüft OpenVINO in Leipzig.**")

        assertEquals("Frau Müller prüft OpenVINO in Leipzig.", result)
    }

    @Test
    fun noteLanguageDetector_prefersGermanUmlautsBeforeFrenchAccents() {
        val result = NoteLanguageDetector.detect("Frau Müller prüft OpenVINO für Leipzig.")

        assertEquals(NoteLanguage.GERMAN, result)
    }

    @Test
    fun noteLanguageDetector_keepsFrenchAccentDetection() {
        val result = NoteLanguageDetector.detect("Claire prépare une revue à Lyon avant mercredi.")

        assertEquals(NoteLanguage.FRENCH, result)
    }

    @Test
    fun normalizeTags_splitsByCommaAndNewLine() {
        val processor = ResultProcessor()

        val result = processor.normalizeTags(" Kotlin, AI\nOpenVINO, kotlin,  ")

        assertEquals(setOf("kotlin", "ai", "openvino"), result)
    }

    @Test
    fun normalizeTags_ignoresBlankItems() {
        val processor = ResultProcessor()

        val result = processor.normalizeTags(",  ,\n  tag-one  ,\n")

        assertEquals(setOf("tag-one"), result)
    }

    @Test
    fun normalizeTags_respectsMaxTags() {
        val processor = ResultProcessor()

        val result = processor.normalizeTags("one, two, three", maxTags = 2)

        assertEquals(setOf("one", "two"), result)
    }

    @Test
    fun normalizeTags_removesLabelsBulletsHashesAndSemicolons() {
        val processor = ResultProcessor()

        val result = processor.normalizeTags("Tags: #OpenVINO; - склад; 2) риск", maxTags = 4)

        assertEquals(setOf("openvino", "склад", "риск"), result)
    }

    @Test
    fun normalizeTags_withSourceDropsUngroundedItemsAndAddsKeywordFallback() {
        val processor = ResultProcessor()

        val result =
            processor.normalizeTags(
                raw = "Tags: maydeterminingfielddemo, OpenVINO, drchendr, labb",
                maxTags = 4,
                sourceText =
                    "Maya is planning a field demo for the hospital logistics robot. " +
                        "She must confirm the OpenVINO build and prepare battery packs.",
            )

        assertEquals(setOf("openvino", "demo", "robot", "battery"), result)
    }

    @Test
    fun normalizeTags_withSourceFiltersLongSentenceTags() {
        val processor = ResultProcessor()

        val result =
            processor.normalizeTags(
                raw = "frau muller qualités prudence halle 2 openvino tempéraux risques",
                maxTags = 4,
                sourceText =
                    "Frau Müller koordiniert eine Qualitätsprüfung in Leipzig. " +
                        "Jonas und Aylin testen die OpenVINO-Auswertung und Temperatursensoren.",
            )

        assertTrue(result.any { it.contains("qualität") })
        assertTrue(result.any { it.contains("openvino") || it.contains("temperatur") })
        assertTrue(result.all { it.length <= 32 })
    }

    @Test
    fun normalizeRewrite_removesGeneratedMetadataSections() {
        val processor = ResultProcessor()

        val result =
            processor.normalizeRewrite(
                """
                Review Android demo on Monday.
                **tags**: android, demo, monday
                **summary**: review demo.
                """.trimIndent(),
            )

        assertEquals("Review Android demo on Monday.", result)
    }

    @Test
    fun normalizeRewrite_removesAssistantLabel() {
        val processor = ResultProcessor()

        val result = processor.normalizeRewrite("Rewritten note: Check Lab B at 14:00.")

        assertEquals("Check Lab B at 14:00.", result)
    }

    @Test
    fun normalizeSummary_withSourceFallsBackWhenGermanAnswerDriftsToFrench() {
        val processor = ResultProcessor()

        val result =
            processor.normalizeSummary(
                raw = "Frau Müller coordonne une qualité en Leipzig.",
                sourceText = "Frau Müller koordiniert eine Qualitätsprüfung in Leipzig. Danach folgt der Test.",
            )

        assertEquals("Frau Müller koordiniert eine Qualitätsprüfung in Leipzig.", result)
    }

    @Test
    fun normalizeSummary_withSourceFallsBackWhenSummaryLoopsTooLong() {
        val processor = ResultProcessor()

        val result =
            processor.normalizeSummary(
                raw = "La qualité de ".repeat(30),
                sourceText = "Frau Müller koordiniert eine Qualitätsprüfung in Leipzig. Danach folgt der Test.",
            )

        assertEquals("Frau Müller koordiniert eine Qualitätsprüfung in Leipzig.", result)
    }

    @Test
    fun normalizeSummary_withSourceFallsBackWhenAnswerIsUnfinished() {
        val processor = ResultProcessor()

        val result =
            processor.normalizeSummary(
                raw = "Марина проверяет пилотный стенд и записывает три риска: задержка поставки, шумные",
                sourceText =
                    "Марина проверяет OpenVINO-стенд в Казани во вторник в 09:30. " +
                        "Иван записывает три риска для команды.",
            )

        assertEquals("Марина проверяет OpenVINO-стенд в Казани во вторник в 09:30.", result)
    }

    @Test
    fun normalizeSummary_withSourceFallsBackWhenAnswerEndsAtAbbreviation() {
        val processor = ResultProcessor()

        val result =
            processor.normalizeSummary(
                raw = "Maya must send Dr.",
                sourceText = "Maya sends Dr. Chen the risk note before Friday 14:00.",
            )

        assertEquals("Maya sends Dr. Chen the risk note before Friday 14:00.", result)
    }

    @Test
    fun normalizeSummary_withSourceRepairsLanguageSpecificArtifacts() {
        val processor = ResultProcessor()

        val german =
            processor.normalizeSummary(
                raw = "Fur die Qualitaetspruefung in Leipzig ist Frau Müller verantwortlich.",
                sourceText = "Frau Müller koordiniert die Qualitätsprüfung in Leipzig.",
            )

        assertEquals("Für die Qualitätsprüfung in Leipzig ist Frau Müller verantwortlich.", german)
    }

    @Test
    fun normalizeSummary_withRussianSourcePrefersExtractiveGrounding() {
        val processor = ResultProcessor()

        val result =
            processor.normalizeSummary(
                raw = "Марина проверяет пилотного стенда в Казани в 09:30.",
                sourceText = "Марина ведет запуск пилотного стенда в Казани в 09:30.",
            )

        assertEquals("Марина ведет запуск пилотного стенда в Казани в 09:30.", result)
    }

    @Test
    fun normalizeRewrite_withSourceRepairsLanguageSpecificArtifacts() {
        val processor = ResultProcessor()

        val result =
            processor.normalizeRewrite(
                raw = "Марина проверяет пилотного стенда в Казани в 09:30.",
                sourceText = "Марина ведет запуск пилотного стенда в Казани в 09:30.",
            )

        assertEquals("Марина проверяет пилотный стенд в Казани в 09:30.", result)
    }

    @Test
    fun normalizeSummary_withSourceCompactsMultiSentenceModelAnswer() {
        val processor = ResultProcessor()

        val result =
            processor.normalizeSummary(
                raw =
                    "Frau Müller koordiniert eine Qualitätsprüfung in Leipzig. " +
                        "Jonas und Aylin testen OpenVINO am Donnerstag um 08:15. " +
                        "Wenn der Lärmpegel steigt, wird der Versuch in Halle 2 verschoben.",
                sourceText =
                    "Frau Müller koordiniert eine Qualitätsprüfung in Leipzig. " +
                        "Am Donnerstag um 08:15 testen Jonas und Aylin OpenVINO in Halle 2. " +
                        "Danach dokumentiert das Team offene Risiken.",
            )

        assertEquals(
            "Frau Müller koordiniert eine Qualitätsprüfung in Leipzig. " +
                "Jonas und Aylin testen OpenVINO am Donnerstag um 08:15.",
            result,
        )
    }

    @Test
    fun normalizeSummary_withSourceFallsBackWhenAnswerAddsGenericClaim() {
        val processor = ResultProcessor()

        val result =
            processor.normalizeSummary(
                raw =
                    "Frau Müller koordiniert eine Qualitätsprüfung in Leipzig, " +
                        "und die Risiken werden in der Regel von der Teamliste abgeleitet.",
                sourceText =
                    "Frau Müller koordiniert eine Qualitätsprüfung in Leipzig. " +
                        "Jonas testet OpenVINO am Donnerstag um 08:15.",
            )

        assertEquals("Jonas testet OpenVINO am Donnerstag um 08:15.", result)
    }

    @Test
    fun normalizeSummary_withSourceSelectsUsefulFallbackFactsAcrossLanguages() {
        val processor = ResultProcessor()
        val cases =
            listOf(
                SummaryFallbackCase(
                    raw = "the model and risk",
                    source = "Заметки к встрече. Марина проверяет OpenVINO-стенд в Казани во вторник в 09:30.",
                    genericFirstSentence = "Заметки к встрече.",
                    expectedFacts = listOf("Марина", "09:30"),
                ),
                SummaryFallbackCase(
                    raw = "Марина проверяет стенд.",
                    source =
                        "Meeting notes. Maya must confirm the OpenVINO build before Friday 14:00 " +
                            "if Lab B is closed.",
                    genericFirstSentence = "Meeting notes.",
                    expectedFacts = listOf("Maya", "14:00"),
                ),
                SummaryFallbackCase(
                    raw = "Frau Müller coordonne une qualité en Leipzig.",
                    source = "Notiz. Frau Müller testet OpenVINO in Leipzig am Donnerstag um 08:15.",
                    genericFirstSentence = "Notiz.",
                    expectedFacts = listOf("Müller", "08:15"),
                ),
                SummaryFallbackCase(
                    raw = "Frau Müller muss prüfen.",
                    source = "Note. Claire vérifie le modèle OpenVINO à Lyon avant mercredi 16:45.",
                    genericFirstSentence = "Note.",
                    expectedFacts = listOf("Claire", "16:45"),
                ),
            )

        cases.forEach { testCase ->
            val result =
                processor.normalizeSummary(
                    raw = testCase.raw,
                    sourceText = testCase.source,
                )

            assertFalse(result, result.equals(testCase.genericFirstSentence, ignoreCase = true))
            testCase.expectedFacts.forEach { fact ->
                assertTrue("$result does not contain $fact", result.contains(fact, ignoreCase = true))
            }
        }
    }

    @Test
    fun normalizeRewrite_withSourceFallsBackWhenGermanAnswerDriftsToFrench() {
        val processor = ResultProcessor()
        val source = "Frau Müller koordiniert eine Qualitätsprüfung in Leipzig."

        val result =
            processor.normalizeRewrite(
                raw = "Frau Müller coordonne une qualité en Leipzig.",
                sourceText = source,
            )

        assertEquals(source, result)
    }

    @Test
    fun normalizeRewrite_withSourceFallsBackWhenAnswerIsUnfinished() {
        val processor = ResultProcessor()
        val source = "Maya confirms the OpenVINO build before Friday 14:00."

        val result =
            processor.normalizeRewrite(
                raw = "Maya confirms the OpenVINO build before Friday 14",
                sourceText = source,
            )

        assertEquals(source, result)
    }

    @Test
    fun normalizeRewrite_withSourceFallbackCleansPunctuationSpacing() {
        val processor = ResultProcessor()

        val result =
            processor.normalizeRewrite(
                raw = "Марина проверяет стенд.",
                sourceText = "Maya   confirms OpenVINO build ;  Friday 14:00 .",
            )

        assertEquals("Maya confirms OpenVINO build; Friday 14:00.", result)
    }

    @Test
    fun normalizeRewrite_withSourceFallsBackWhenPromptInstructionIsEchoed() {
        val processor = ResultProcessor()
        val source = "Discuss roadmap openvino tasks, groceries, and call Bob."

        val result =
            processor.normalizeRewrite(
                raw = "Start immediately with the rewritten note.",
                sourceText = source,
            )

        assertEquals(source, result)
    }

    @Test
    fun normalizeRewrite_withSourceFallsBackWhenRequiredFactsAreLost() {
        val processor = ResultProcessor()
        val source = "Discuss roadmap openvino tasks, groceries, and call Bob."

        val result =
            processor.normalizeRewrite(
                raw = "Start the work soon.",
                sourceText = source,
            )

        assertEquals(source, result)
    }

    private data class SummaryFallbackCase(
        val raw: String,
        val source: String,
        val genericFirstSentence: String,
        val expectedFacts: List<String>,
    )
}

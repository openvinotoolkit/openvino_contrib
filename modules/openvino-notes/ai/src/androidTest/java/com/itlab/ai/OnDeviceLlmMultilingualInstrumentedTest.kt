package com.itlab.ai

import android.os.SystemClock
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.itlab.domain.ai.RewriteStyle
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class OnDeviceLlmMultilingualInstrumentedTest {
    @Test(timeout = 600_000)
    fun generatesUsefulAnswersForComplexMultilingualNotes() =
        runBlocking {
            val fixture = createFixture()
            try {
                val warmUpMs = measureElapsedMs { fixture.service.warmUp() }
                val warmDiagnostics = fixture.backend.diagnostics()
                assertEquals(
                    "Warm-up must create exactly one reusable pipeline",
                    1,
                    warmDiagnostics.pipelineCreationCount,
                )
                assertTrue("OpenVINO cache directory was not created", warmDiagnostics.cacheDir.isDirectory)
                logInfo(
                    "warmUpMs=$warmUpMs cacheDir=${warmDiagnostics.cacheDir} " +
                        "cacheFiles=${warmDiagnostics.cacheFileCount}",
                )

                val timings = mutableListOf<GenerationTiming>()
                multilingualCases.forEach { testCase ->
                    val before = fixture.backend.diagnostics()
                    assertEquals("Pipeline was recreated before ${testCase.language}", 1, before.pipelineCreationCount)

                    val summaryResult =
                        measureResult {
                            fixture.service.summarize(
                                text = testCase.note,
                                maxInputTokens = config.summaryMaxInputTokens,
                                maxNewTokens = config.summaryMaxNewTokens,
                            )
                        }
                    val tagsResult =
                        measureResult {
                            fixture.service.suggestTags(
                                text = testCase.note,
                                maxInputTokens = config.tagsMaxInputTokens,
                                maxTags = config.maxTags,
                            )
                        }
                    val rewriteResult =
                        measureResult {
                            fixture.service.rewrite(
                                text = testCase.note,
                                style = RewriteStyle.CLEANUP,
                                maxInputTokens = config.rewriteMaxInputTokens,
                                maxNewTokens = config.rewriteMaxNewTokens,
                            )
                        }

                    val after = fixture.backend.diagnostics()
                    assertEquals("Pipeline was recreated during ${testCase.language}", 1, after.pipelineCreationCount)

                    timings += GenerationTiming(testCase.language, "summary", summaryResult.elapsedMs)
                    timings += GenerationTiming(testCase.language, "tags", tagsResult.elapsedMs)
                    timings += GenerationTiming(testCase.language, "rewrite", rewriteResult.elapsedMs)

                    logInfo(
                        "${testCase.language} " +
                            "summaryMs=${summaryResult.elapsedMs} tagsMs=${tagsResult.elapsedMs} " +
                            "rewriteMs=${rewriteResult.elapsedMs} " +
                            "summaryChars=${summaryResult.value.length} " +
                            "tagsCount=${tagsResult.value.size} " +
                            "rewriteChars=${rewriteResult.value.length}",
                    )
                    logInfo("${testCase.language} summary=${summaryResult.value}")
                    logInfo("${testCase.language} tags=${tagsResult.value.joinToString()}")
                    logInfo("${testCase.language} rewrite=${rewriteResult.value}")

                    assertUsefulSummary(testCase, summaryResult.value)
                    assertUsefulTags(testCase, tagsResult.value)
                    assertUsefulRewrite(testCase, rewriteResult.value)
                }

                assertWarmGenerationPerformance(timings)
            } finally {
                fixture.service.release()
            }
        }

    private fun createFixture(): TestFixture {
        val context = InstrumentationRegistry.getInstrumentation().targetContext.applicationContext
        val backend = OpenVinoGenAiBackend(context, config)
        val service =
            OpenVinoNoteAiService(
                engine =
                    OpenVinoEngine(
                        llmBackend = backend,
                        promptBuilder = NoteLlmPromptBuilder(config),
                        config = config,
                    ),
                processor = ResultProcessor(),
            )
        return TestFixture(
            backend = backend,
            service = service,
        )
    }

    private inline fun measureElapsedMs(block: () -> Unit): Long {
        val startedAt = SystemClock.elapsedRealtime()
        block()
        return SystemClock.elapsedRealtime() - startedAt
    }

    private inline fun <T> measureResult(block: () -> T): TimedResult<T> {
        val startedAt = SystemClock.elapsedRealtime()
        val value = block()
        return TimedResult(
            value = value,
            elapsedMs = SystemClock.elapsedRealtime() - startedAt,
        )
    }

    private fun assertUsefulSummary(
        testCase: MultilingualCase,
        summary: String,
    ) {
        assertTrue("${testCase.language} summary is blank", summary.isNotBlank())
        assertTrue("${testCase.language} summary is too long: $summary", summary.length <= 260)
        assertTrue(
            "${testCase.language} summary is not compressed enough: $summary",
            summary.length <= testCase.note.length * 2 / 3,
        )
        assertCompleteAnswer(testCase.language, "summary", summary)
        assertFalse(
            "${testCase.language} summary copied the source note: $summary",
            normalizedForComparison(summary) == normalizedForComparison(testCase.note),
        )
        assertNoAssistantArtifacts(testCase.language, "summary", summary)
        assertLanguageSignal(testCase, "summary", summary)
        assertContainsAtLeast(testCase, "summary", summary, testCase.summaryFacts, minimumMatches = 2)
    }

    private fun assertUsefulTags(
        testCase: MultilingualCase,
        tags: Set<String>,
    ) {
        assertTrue("${testCase.language} tags are blank", tags.isNotEmpty())
        assertTrue("${testCase.language} too many tags: $tags", tags.size <= config.maxTags)
        tags.forEach { tag ->
            assertFalse("${testCase.language} tag contains markdown: $tag", tag.contains("*"))
            assertFalse("${testCase.language} tag contains prompt artifact: $tag", tag.contains("<|"))
            assertTrue("${testCase.language} tag is too long: $tag", tag.length <= 32)
        }
        assertTrue(
            "${testCase.language} tags do not contain expected language/topic signal: $tags",
            testCase.tagSignals.any { signal ->
                tags.any { tag -> tag.contains(signal, ignoreCase = true) }
            },
        )
    }

    private fun assertUsefulRewrite(
        testCase: MultilingualCase,
        rewrite: String,
    ) {
        assertTrue("${testCase.language} rewrite is blank", rewrite.isNotBlank())
        assertTrue("${testCase.language} rewrite is too short: $rewrite", rewrite.length >= 80)
        assertTrue(
            "${testCase.language} rewrite is unexpectedly long: $rewrite",
            rewrite.length <= testCase.note.length + 220,
        )
        assertNoAssistantArtifacts(testCase.language, "rewrite", rewrite)
        assertCompleteAnswer(testCase.language, "rewrite", rewrite)
        assertLanguageSignal(testCase, "rewrite", rewrite)
        assertContainsAtLeast(testCase, "rewrite", rewrite, testCase.rewriteFacts, minimumMatches = 2)
    }

    private fun assertNoAssistantArtifacts(
        language: String,
        field: String,
        value: String,
    ) {
        assertFalse("$language $field contains chat template artifact: $value", value.contains("<|"))
        assertFalse("$language $field contains markdown heading: $value", value.contains("**"))
    }

    private fun assertCompleteAnswer(
        language: String,
        field: String,
        value: String,
    ) {
        assertTrue(
            "$language $field is unfinished: $value",
            value.trim().lastOrNull() in terminalPunctuation,
        )
    }

    private fun assertLanguageSignal(
        testCase: MultilingualCase,
        field: String,
        value: String,
    ) {
        assertTrue(
            "${testCase.language} $field does not preserve the expected language signal: $value",
            testCase.languageSignal.containsMatchIn(value),
        )
    }

    private fun assertContainsAtLeast(
        testCase: MultilingualCase,
        field: String,
        value: String,
        expectedFacts: List<String>,
        minimumMatches: Int,
    ) {
        val matchedFacts = expectedFacts.filter { fact -> value.contains(fact, ignoreCase = true) }
        assertTrue(
            "${testCase.language} $field lost key facts. Expected at least $minimumMatches of $expectedFacts in: $value",
            matchedFacts.size >= minimumMatches,
        )
    }

    private fun normalizedForComparison(value: String): String =
        value
            .lowercase()
            .replace(Regex("""\s+"""), " ")
            .trim()

    private fun assertWarmGenerationPerformance(timings: List<GenerationTiming>) {
        val slowest = timings.maxBy { it.elapsedMs }
        val averageMs = timings.sumOf { it.elapsedMs } / timings.size
        logInfo("warm generation timings=$timings averageMs=$averageMs slowest=$slowest")
        assertTrue(
            "Warm LLM generation average is too slow: ${averageMs}ms, timings=$timings",
            averageMs <= MAX_AVERAGE_WARM_GENERATION_MS,
        )
        assertTrue(
            "A warm LLM generation call is too slow: $slowest, timings=$timings",
            slowest.elapsedMs <= MAX_SINGLE_WARM_GENERATION_MS,
        )
    }

    private fun logInfo(message: String) {
        if (Log.isLoggable(TAG, Log.INFO)) {
            Log.i(TAG, message)
        }
    }

    private data class TestFixture(
        val backend: OpenVinoGenAiBackend,
        val service: OpenVinoNoteAiService,
    )

    private data class TimedResult<T>(
        val value: T,
        val elapsedMs: Long,
    )

    private data class GenerationTiming(
        val language: String,
        val action: String,
        val elapsedMs: Long,
    )

    private data class MultilingualCase(
        val language: String,
        val note: String,
        val languageSignal: Regex,
        val summaryFacts: List<String>,
        val rewriteFacts: List<String>,
        val tagSignals: List<String>,
    )

    private companion object {
        private const val TAG = "OnDeviceLlmTest"
        private const val MAX_AVERAGE_WARM_GENERATION_MS = 18_000L
        private const val MAX_SINGLE_WARM_GENERATION_MS = 45_000L
        private val terminalPunctuation = setOf('.', '!', '?', '。', '！', '？')
        private val config = OnDeviceLlmConfig.defaultAndroid()

        private val multilingualCases =
            listOf(
                MultilingualCase(
                    language = "Russian",
                    note =
                        "Марина ведет запуск пилотного стенда для умного склада. Во вторник в 09:30 " +
                            "нужно проверить OpenVINO-модель, датчики влажности и отчет для команды в Казани. " +
                            "Если температура выше 28 °C, Иван должен перенести коробки с лекарствами в холодную " +
                            "зону. Еще надо записать три риска: задержка поставки, шумные показания и нехватка батарей.",
                    languageSignal = Regex("[А-Яа-яЁё]"),
                    summaryFacts = listOf("Марина", "Иван", "Казан", "09:30", "28"),
                    rewriteFacts = listOf("Марина", "Иван", "Казан", "09:30", "28"),
                    tagSignals = listOf("склад", "датчик", "риск", "модель", "казан", "стенд", "запуск", "отчет"),
                ),
                MultilingualCase(
                    language = "English",
                    note =
                        "Maya is planning a field demo for the hospital logistics robot. Before Friday 14:00, " +
                            "she must confirm the OpenVINO build, prepare two battery packs, and send Dr. Chen " +
                            "a risk note about the elevator outage. If the rain forecast changes, move the demo " +
                            "from the west entrance to Lab B.",
                    languageSignal =
                        Regex(
                            "\\b(the|and|with|for|before|after|must|should|needs?)\\b",
                            RegexOption.IGNORE_CASE,
                        ),
                    summaryFacts = listOf("Maya", "Chen", "Friday", "14:00", "Lab B"),
                    rewriteFacts = listOf("Maya", "Chen", "Friday", "14:00", "Lab B"),
                    tagSignals = listOf("robot", "hospital", "demo", "battery", "risk"),
                ),
                MultilingualCase(
                    language = "German",
                    note =
                        "Frau Müller koordiniert eine Qualitätsprüfung in Leipzig. Am Donnerstag um 08:15 " +
                            "sollen Jonas und Aylin die OpenVINO-Auswertung, drei Temperatursensoren und den " +
                            "Ersatzakku testen. Wenn der Lärmpegel über 70 dB steigt, wird der Versuch in Halle 2 " +
                            "verschoben. Danach braucht das Team eine kurze Liste der offenen Risiken.",
                    languageSignal =
                        Regex(
                            "\\b(der|die|das|und|mit|für|bis|soll|sollen|muss|am|im|eine|den)\\b|[äöüß]",
                            RegexOption.IGNORE_CASE,
                        ),
                    summaryFacts = listOf("Müller", "Leipzig", "08:15", "70", "Halle 2"),
                    rewriteFacts = listOf("Müller", "Leipzig", "08:15", "70", "Halle 2"),
                    tagSignals = listOf("qualität", "temperatur", "risik", "leipzig", "halle"),
                ),
                MultilingualCase(
                    language = "French",
                    note =
                        "Claire prépare une revue de prototype à Lyon. Avant mercredi 16:45, elle doit vérifier " +
                            "le modèle OpenVINO, comparer quatre mesures de pression et envoyer à Marc une synthèse " +
                            "des risques. Si le fournisseur retarde la vanne, la réunion passe en salle Atlas et le " +
                            "test sécurité devient prioritaire.",
                    languageSignal =
                        Regex(
                            "\\b(le|la|les|des|et|avec|pour|avant|doit|une|en)\\b|[àâçéèêëîïôûùüÿœ]",
                            RegexOption.IGNORE_CASE,
                        ),
                    summaryFacts = listOf("Claire", "Lyon", "16:45", "Marc", "Atlas"),
                    rewriteFacts = listOf("Claire", "Lyon", "16:45", "Marc", "Atlas"),
                    tagSignals = listOf("risque", "pression", "prototype", "lyon", "sécurité"),
                ),
            )
    }
}

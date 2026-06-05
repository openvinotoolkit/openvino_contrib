package com.itlab.ai

import com.itlab.domain.ai.RewriteStyle
import com.ovx.openvino.genai.AndroidPipelineProperties
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.io.path.createTempDirectory

class OpenVinoAiLayerTest {
    @Test
    fun summarize_returnsTrimmedSummary() =
        runBlocking {
            val backend = RecordingLlmBackend("  Summary text.  ")
            val service =
                OpenVinoNoteAiService(
                    OpenVinoEngine(llmBackend = backend),
                    ResultProcessor(),
                )

            val result =
                service.summarize(
                    text = "Long note",
                    maxInputTokens = OnDeviceLlmConfig.defaultAndroid().summaryMaxInputTokens,
                    maxNewTokens = OnDeviceLlmConfig.defaultAndroid().summaryMaxNewTokens,
                )

            assertEquals("Summary text.", result)
            assertEquals(OnDeviceLlmConfig.defaultAndroid().summaryMaxNewTokens, backend.lastMaxNewTokens)
            assertEquals(LlmGenerationIntent.Summary, backend.lastIntent)
            assertTrue(backend.lastPrompt.orEmpty().contains("Summarize the note"))
            assertTrue(backend.lastPrompt.orEmpty().contains("Detected note language: English"))
            assertTrue(backend.lastPrompt.orEmpty().contains("Answer only in English"))
            assertTrue(backend.lastPrompt.orEmpty().contains("Do not translate names"))
            assertTrue(backend.lastPrompt.orEmpty().contains("Long note"))
        }

    @Test
    fun summarize_retriesInvalidModelAnswerBeforeFallback() =
        runBlocking {
            val backend =
                RecordingLlmBackend(
                    "Frau Müller coordonne une qualité en Leipzig.",
                    "Frau Müller koordiniert eine Qualitätsprüfung in Leipzig.",
                )
            val service =
                OpenVinoNoteAiService(
                    OpenVinoEngine(llmBackend = backend),
                    ResultProcessor(),
                )

            val result =
                service.summarize(
                    text = "Frau Müller koordiniert eine Qualitätsprüfung in Leipzig.",
                    maxInputTokens = OnDeviceLlmConfig.defaultAndroid().summaryMaxInputTokens,
                    maxNewTokens = OnDeviceLlmConfig.defaultAndroid().summaryMaxNewTokens,
                )

            assertEquals("Frau Müller koordiniert eine Qualitätsprüfung in Leipzig.", result)
            assertEquals(2, backend.generateCallCount)
            assertTrue(backend.prompts[1].contains("previous summary was invalid", ignoreCase = true))
        }

    @Test
    fun suggestTags_normalizesCaseAndSeparators() =
        runBlocking {
            val backend = RecordingLlmBackend(" Kotlin, Notes\nAI ")
            val service =
                OpenVinoNoteAiService(
                    OpenVinoEngine(llmBackend = backend),
                    ResultProcessor(),
                )

            val result =
                service.suggestTags(
                    text = "Kotlin Notes AI OpenVINO note",
                    maxInputTokens = OnDeviceLlmConfig.defaultAndroid().tagsMaxInputTokens,
                    maxTags = 3,
                )

            assertEquals(setOf("kotlin", "notes", "ai"), result)
            assertEquals(OnDeviceLlmConfig.defaultAndroid().tagsMaxNewTokens, backend.lastMaxNewTokens)
            assertEquals(LlmGenerationIntent.Tags, backend.lastIntent)
            assertTrue(backend.lastPrompt.orEmpty().contains("Suggest up to"))
            assertTrue(backend.lastPrompt.orEmpty().contains("Prefer tags in English"))
            assertTrue(backend.lastPrompt.orEmpty().contains("OpenVINO note"))
        }

    @Test
    fun suggestTags_retriesWhenGeneratedTagsAreNotGrounded() =
        runBlocking {
            val backend =
                RecordingLlmBackend(
                    "frau muller qualités prudence halle 2 openvino tempéraux risques",
                    "Qualitätsprüfung, OpenVINO, Leipzig",
                )
            val service =
                OpenVinoNoteAiService(
                    OpenVinoEngine(llmBackend = backend),
                    ResultProcessor(),
                )

            val result =
                service.suggestTags(
                    text = "Frau Müller koordiniert eine Qualitätsprüfung in Leipzig mit OpenVINO.",
                    maxInputTokens = OnDeviceLlmConfig.defaultAndroid().tagsMaxInputTokens,
                    maxTags = 3,
                )

            assertEquals(setOf("qualitätsprüfung", "openvino", "leipzig"), result)
            assertEquals(2, backend.generateCallCount)
            assertTrue(backend.prompts[1].contains("previous tag list was invalid", ignoreCase = true))
        }

    @Test
    fun rewrite_returnsTrimmedRewrite() =
        runBlocking {
            val backend = RecordingLlmBackend("  Better note.  ")
            val service =
                OpenVinoNoteAiService(
                    OpenVinoEngine(llmBackend = backend),
                    ResultProcessor(),
                )

            val result =
                service.rewrite(
                    text = "Draft note",
                    style = RewriteStyle.CLEANUP,
                    maxInputTokens = OnDeviceLlmConfig.defaultAndroid().rewriteMaxInputTokens,
                    maxNewTokens = OnDeviceLlmConfig.defaultAndroid().rewriteMaxNewTokens,
                )

            assertEquals("Better note.", result)
            assertEquals(OnDeviceLlmConfig.defaultAndroid().rewriteMaxNewTokens, backend.lastMaxNewTokens)
            assertEquals(LlmGenerationIntent.Rewrite, backend.lastIntent)
            assertTrue(backend.lastPrompt.orEmpty().contains("Rewrite the note"))
            assertTrue(backend.lastPrompt.orEmpty().contains("Do not summarize or translate"))
            assertFalse(backend.lastPrompt.orEmpty().contains("Start immediately with the rewritten note"))
            assertTrue(backend.lastPrompt.orEmpty().contains("Draft note"))
        }

    @Test
    fun rewrite_retriesInvalidModelAnswerBeforeFallback() =
        runBlocking {
            val backend =
                RecordingLlmBackend(
                    "Frau Müller coordonne une qualité en Leipzig.",
                    "Frau Müller koordiniert eine Qualitätsprüfung in Leipzig.",
                )
            val service =
                OpenVinoNoteAiService(
                    OpenVinoEngine(llmBackend = backend),
                    ResultProcessor(),
                )

            val result =
                service.rewrite(
                    text = "Frau Müller koordiniert eine Qualitätsprüfung in Leipzig.",
                    style = RewriteStyle.CLEANUP,
                    maxInputTokens = OnDeviceLlmConfig.defaultAndroid().rewriteMaxInputTokens,
                    maxNewTokens = OnDeviceLlmConfig.defaultAndroid().rewriteMaxNewTokens,
                )

            assertEquals("Frau Müller koordiniert eine Qualitätsprüfung in Leipzig.", result)
            assertEquals(2, backend.generateCallCount)
            assertTrue(backend.prompts[1].contains("previous rewrite was invalid", ignoreCase = true))
        }

    @Test
    fun rewrite_retriesPromptEchoAndDoesNotApplyBadRetry() =
        runBlocking {
            val source = "Discuss roadmap openvino tasks, groceries, and call Bob."
            val backend =
                RecordingLlmBackend(
                    "Start immediately with the rewritten note.",
                    "Start the work soon.",
                )
            val service =
                OpenVinoNoteAiService(
                    OpenVinoEngine(llmBackend = backend),
                    ResultProcessor(),
                )

            val result =
                service.rewrite(
                    text = source,
                    style = RewriteStyle.CLEANUP,
                    maxInputTokens = OnDeviceLlmConfig.defaultAndroid().rewriteMaxInputTokens,
                    maxNewTokens = OnDeviceLlmConfig.defaultAndroid().rewriteMaxNewTokens,
                )

            assertEquals(source, result)
            assertEquals(2, backend.generateCallCount)
            assertFalse(backend.prompts[1].contains("Start immediately with the rewritten note"))
        }

    @Test
    fun tagIMGs_delegatesToImageTaggingBackend() =
        runBlocking {
            val imageBackend = RecordingImageTaggingBackend(setOf("bus", "person"))
            val service =
                OpenVinoNoteAiService(
                    OpenVinoEngine(llmBackend = RecordingLlmBackend("unused")),
                    ResultProcessor(),
                    imageTaggingBackend = imageBackend,
                )

            val result = service.tagIMGs(listOf("/tmp/bus.jpg"))

            assertEquals(setOf("bus", "person"), result)
            assertEquals(listOf("/tmp/bus.jpg"), imageBackend.lastSources)
        }

    @Test
    fun yoloOutputParser_returnsTagsWithClassAwareNmsAndConfidenceFilter() {
        val config =
            OnDeviceVisionConfig
                .defaultAndroid()
                .copy(confidenceThreshold = 0.35f, iouThreshold = 0.45f, maxDetections = 4)
        val parser = YoloOutputParser(config)
        val outputData =
            floatArrayOf(
                0f,
                0f,
                10f,
                10f,
                0.90f,
                0f,
                1f,
                1f,
                11f,
                11f,
                0.80f,
                1f,
                20f,
                20f,
                30f,
                30f,
                0.34f,
                2f,
                40f,
                40f,
                50f,
                50f,
                0.70f,
                3f,
            )

        val result = parser.parseTags(outputData, listOf("bus", "person", "car", "traffic light"))

        assertEquals(setOf("bus", "person", "traffic light"), result)
    }

    @Test
    fun onDeviceVisionModelSelector_usesStandardModelForCapableDevices() {
        val model =
            OnDeviceVisionModelSelector.select(
                OnDeviceVisionConfig.defaultAndroid(),
                OnDeviceVisionDeviceProfile(cpuCores = 8, totalRamMb = 4096),
            )

        assertEquals("standard", model.id)
    }

    @Test
    fun onDeviceVisionModelSelector_usesCompactModelForWeakDevices() {
        val model =
            OnDeviceVisionModelSelector.select(
                OnDeviceVisionConfig.defaultAndroid(),
                OnDeviceVisionDeviceProfile(cpuCores = 2, totalRamMb = 1024),
            )

        assertEquals("compact", model.id)
    }

    @Test
    fun onDeviceVisionModelSelector_respectsExplicitModelPreference() {
        val model =
            OnDeviceVisionModelSelector.select(
                OnDeviceVisionConfig.defaultAndroid().copy(preferredModelId = "compact"),
                OnDeviceVisionDeviceProfile(cpuCores = 8, totalRamMb = 4096),
            )

        assertEquals("compact", model.id)
    }

    @Test
    fun noteLlmPromptBuilder_trimsLargeInput() {
        val config = OnDeviceLlmConfig.defaultAndroid().copy(maxInputChars = 5)
        val builder = NoteLlmPromptBuilder(config)

        val prompt = builder.summaryPrompt("123456789")

        assertTrue(prompt.contains("12345"))
        assertTrue(!prompt.contains("123456"))
    }

    @Test
    fun noteLlmPromptBuilder_keepsComplexMultilingualFactsInPrompts() {
        val builder = NoteLlmPromptBuilder()
        val notes =
            listOf(
                "Марина проверяет стенд в Казани во вторник в 09:30.",
                "Maya sends Dr. Chen the Lab B risk note before Friday 14:00.",
                "Frau Müller prüft in Leipzig um 08:15 den Akku.",
                "Claire prépare une revue à Lyon avant mercredi 16:45.",
            )

        notes.forEach { note ->
            val summaryPrompt = builder.summaryPrompt(note)
            val tagsPrompt = builder.tagsPrompt(note)
            val rewritePrompt = builder.rewritePrompt(note, RewriteStyle.CLEANUP)

            assertTrue(summaryPrompt.contains(note))
            assertTrue(tagsPrompt.contains(note))
            assertTrue(rewritePrompt.contains(note))
            assertTrue(summaryPrompt.contains("Answer only in"))
            assertTrue(rewritePrompt.contains("Preserve every named person"))
        }
    }

    @Test
    fun androidPipelineProperties_enablesAndroidCacheAndFastCpuSettings() {
        val cacheDir = createTempDirectory("openvino-genai-cache").toFile()
        val config =
            OnDeviceLlmConfig.defaultAndroid().copy(
                inferenceNumThreads = 4,
                numStreams = 1,
                kvCachePrecision = "u8",
                dynamicQuantizationGroupSize = 32,
            )

        val properties =
            AndroidPipelineProperties
                .cpuLatency(
                    cacheDir,
                    AndroidPipelineProperties
                        .CpuLatencyOptions
                        .builder()
                        .numStreams(config.numStreams.toLong())
                        .inferenceNumThreads(config.inferenceNumThreads.toLong())
                        .kvCachePrecision(config.kvCachePrecision)
                        .dynamicQuantizationGroupSize(config.dynamicQuantizationGroupSize.toLong())
                        .build(),
                ).toMap()

        assertEquals(cacheDir.absolutePath, properties["CACHE_DIR"])
        assertEquals("SDPA", properties["ATTENTION_BACKEND"])
        assertEquals("LATENCY", properties["PERFORMANCE_HINT"])
        assertEquals(true, properties["ENABLE_MMAP"])
        assertEquals(1L, properties["NUM_STREAMS"])
        assertEquals(4L, properties["INFERENCE_NUM_THREADS"])
        assertFalse(properties.containsKey("MAX_PROMPT_LEN"))
        assertFalse(properties.containsKey("MIN_RESPONSE_LEN"))
        assertEquals("u8", properties["KV_CACHE_PRECISION"])
        assertEquals(32L, properties["DYNAMIC_QUANTIZATION_GROUP_SIZE"])
    }

    @Test
    fun defaultAndroidInferenceThreads_usesMoreThanOneThreadOnModernPhones() {
        assertEquals(4, OnDeviceLlmConfig.defaultAndroidInferenceThreads(8))
        assertEquals(3, OnDeviceLlmConfig.defaultAndroidInferenceThreads(4))
        assertEquals(1, OnDeviceLlmConfig.defaultAndroidInferenceThreads(1))
    }

    @Test
    fun openVinoEngine_returnsEmptyResultForBlankInputWithoutCallingBackend() {
        val backend = RecordingLlmBackend("unused")
        val engine = OpenVinoEngine(llmBackend = backend)

        assertEquals("", engine.runLlmSummary("   "))
        assertEquals("", engine.runLlmTagging("\n\t"))
        assertEquals("", engine.runLlmRewrite(" ", RewriteStyle.CLEANUP))
        assertEquals(null, backend.lastPrompt)
    }

    @Test
    fun openVinoEngine_releasesBackend() {
        val backend = RecordingLlmBackend("unused")
        val engine = OpenVinoEngine(llmBackend = backend)

        engine.release()

        assertTrue(backend.releaseCalled)
    }

    @Test
    fun unavailableBackend_failsWithConfiguredReason() {
        val backend = UnavailableLlmBackend("missing runtime")

        val failure =
            runCatching {
                backend.generate("prompt", maxNewTokens = 1)
            }.exceptionOrNull()

        assertTrue(failure is MissingLlmRuntimeException)
        assertEquals("missing runtime", failure?.message)
    }

    @Test
    fun missingRuntimeException_preservesCause() {
        val cause = IllegalArgumentException("native loader")

        val failure = MissingLlmRuntimeException("runtime failed", cause)

        assertEquals("runtime failed", failure.message)
        assertEquals(cause, failure.cause)
    }

    @Test
    fun preparePrompt_appendsNoThinkHintWhenReasoningOutputIsDisabled() {
        val config =
            OnDeviceLlmConfig.defaultAndroid().copy(
                includeReasoningOutput = false,
                disableReasoningPromptHint = "/no_think",
            )

        val prompt = preparePrompt("Say ok   ", config)

        assertEquals("Say ok\n/no_think", prompt)
    }

    @Test
    fun preparePrompt_keepsPromptWhenHintAlreadyExists() {
        val config = OnDeviceLlmConfig.defaultAndroid()

        val prompt = preparePrompt("Say ok\n/NO_THINK", config)

        assertEquals("Say ok\n/NO_THINK", prompt)
    }

    @Test
    fun preparePrompt_keepsPromptWhenReasoningOutputIsEnabled() {
        val config =
            OnDeviceLlmConfig.defaultAndroid().copy(
                includeReasoningOutput = true,
                disableReasoningPromptHint = "/no_think",
            )

        val prompt = preparePrompt("Say ok", config)

        assertEquals("Say ok", prompt)
    }

    @Test
    fun preparePrompt_keepsBlankPrompt() {
        val config = OnDeviceLlmConfig.defaultAndroid()

        val prompt = preparePrompt("   ", config)

        assertEquals("   ", prompt)
    }

    @Test
    fun stripReasoningSections_removesThinkingBlockAndTags() {
        val response =
            """
            <think>
            hidden chain
            </think>

            ok
            """.trimIndent()

        val cleaned = stripReasoningSections(response)

        assertEquals("ok", cleaned)
        assertFalse(cleaned.contains("think", ignoreCase = true))
    }

    @Test
    fun stripReasoningSections_removesDanglingThinkingTags() {
        val cleaned = stripReasoningSections("<think>ok")

        assertEquals("ok", cleaned)
    }

    @Test
    fun stripReasoningSections_keepsBlankResponse() {
        val response = "   "

        val cleaned = stripReasoningSections(response)

        assertEquals(response, cleaned)
    }

    private class RecordingLlmBackend(
        private vararg val responses: String,
    ) : LlmInferenceBackend {
        var lastPrompt: String? = null
            private set
        var lastMaxNewTokens: Int? = null
            private set
        var lastIntent: LlmGenerationIntent? = null
            private set
        var releaseCalled: Boolean = false
            private set
        val prompts = mutableListOf<String>()
        val generateCallCount: Int
            get() = prompts.size

        override fun generate(
            prompt: String,
            maxNewTokens: Int,
            intent: LlmGenerationIntent,
        ): String {
            lastPrompt = prompt
            lastMaxNewTokens = maxNewTokens
            lastIntent = intent
            prompts += prompt
            val responseIndex = (generateCallCount - 1).coerceAtMost(responses.lastIndex)
            return responses.getOrElse(responseIndex) { "" }
        }

        override fun release() {
            releaseCalled = true
        }
    }

    private class RecordingImageTaggingBackend(
        private val tags: Set<String>,
    ) : ImageTaggingBackend {
        var lastSources: List<String> = emptyList()
            private set

        override suspend fun tagImages(imageSources: List<String>): Set<String> {
            lastSources = imageSources
            return tags
        }
    }
}

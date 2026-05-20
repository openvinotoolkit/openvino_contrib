package com.itlab.ai

import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Test

class OpenVinoAiLayerTest {
    @Test
    fun normalizeSummary_trimsSpaces() {
        val processor = ResultProcessor()

        val result = processor.normalizeSummary("  short summary  ")

        assertEquals("short summary", result)
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
    fun summarize_returnsTrimmedSummary() =
        runBlocking {
            val service = OpenVinoNoteAiService(OpenVinoEngine(), ResultProcessor())

            val result = service.summarize("  Summary text  ")

            assertEquals("Summary text", result)
        }

    @Test
    fun tagTXT_normalizesCaseAndSeparators() =
        runBlocking {
            val service = OpenVinoNoteAiService(OpenVinoEngine(), ResultProcessor())

            val result = service.tagTXT(" Kotlin, Notes\nAI ")

            assertEquals(setOf("kotlin", "notes", "ai"), result)
        }

    @Test
    fun tagIMGs_aggregatesAndDeduplicatesTags() =
        runBlocking {
            val service = OpenVinoNoteAiService(OpenVinoEngine(), ResultProcessor())

            val result = service.tagIMGs(listOf("Cat, Pet", "pet, animal", "  CAT"))

            assertEquals(setOf("cat", "pet", "animal"), result)
        }
}

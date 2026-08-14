// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package com.itlab.assistant.core

import com.itlab.ai.image.api.FakeImageTagger
import com.itlab.ai.text.api.FakeTextAssistant
import com.itlab.ai.text.api.TextOutcome
import com.itlab.assistant.api.SuggestionKind
import com.itlab.assistant.api.SuggestionOutcome
import com.itlab.identity.api.AccountId
import com.itlab.notes.api.FakeNotesService
import com.itlab.notes.api.Note
import com.itlab.notes.api.NoteId
import java.time.Instant
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class DefaultNoteAssistantTest {
    @Test fun `title suggestion is routed through text API`() = runTest {
        val note = Note(NoteId("note"), AccountId("account"), "Old", "Long body", updatedAt = Instant.EPOCH)
        val text = FakeTextAssistant().apply { outcome = TextOutcome.Generated("New title") }
        val assistant = DefaultNoteAssistant(FakeNotesService(listOf(note)), text, FakeImageTagger())

        val outcome = assistant.suggestTitle(note.id)

        assertTrue(outcome is SuggestionOutcome.Ready)
        assertEquals(SuggestionKind.TITLE, (outcome as SuggestionOutcome.Ready).suggestion.kind)
        assertEquals("New title", outcome.suggestion.value)
    }
}

// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package com.openvino.notes.view.assistant

import com.openvino.notes.assistant.api.ApplySuggestionOutcome
import com.openvino.notes.assistant.api.AssistantRewriteStyle
import com.openvino.notes.assistant.api.AssistantSuggestion
import com.openvino.notes.assistant.api.NoteAssistant
import com.openvino.notes.assistant.api.SuggestionOutcome
import com.openvino.notes.notes.api.AttachmentId
import com.openvino.notes.notes.api.ContentItemId
import com.openvino.notes.notes.api.NoteId
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.test.StandardTestDispatcher
import kotlinx.coroutines.test.advanceUntilIdle
import kotlinx.coroutines.test.resetMain
import kotlinx.coroutines.test.runTest
import kotlinx.coroutines.test.setMain
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test

@OptIn(ExperimentalCoroutinesApi::class)
class AssistantViewModelTest {

    private val testDispatcher = StandardTestDispatcher()

    private class FakeUnavailableAssistant : NoteAssistant {
        var applyCallsCount = 0

        override suspend fun summarize(noteId: NoteId): SuggestionOutcome {
            return SuggestionOutcome.Unavailable("Inference models are not configured")
        }

        override suspend fun suggestTextTags(noteId: NoteId): SuggestionOutcome {
            return SuggestionOutcome.Unavailable("Inference models are not configured")
        }

        override suspend fun rewrite(
            noteId: NoteId,
            contentItemId: ContentItemId,
            style: AssistantRewriteStyle,
        ): SuggestionOutcome {
            return SuggestionOutcome.Unavailable("Inference models are not configured")
        }

        override suspend fun suggestImageTags(
            noteId: NoteId,
            attachmentId: AttachmentId,
        ): SuggestionOutcome {
            return SuggestionOutcome.Unavailable("Inference models are not configured")
        }

        override suspend fun apply(suggestion: AssistantSuggestion): ApplySuggestionOutcome {
            applyCallsCount++
            return ApplySuggestionOutcome.Applied
        }
    }

    @Before
    fun setUp() {
        Dispatchers.setMain(testDispatcher)
    }

    @After
    fun tearDown() {
        Dispatchers.resetMain()
    }

    @Test
    fun initialStateExplainsUnavailability() {
        val fakeAssistant = FakeUnavailableAssistant()
        val viewModel = AssistantViewModel(fakeAssistant)

        val state = viewModel.state.value
        assertFalse(state.isAvailable)
        assertFalse(state.running)
        assertNotNull(state.unavailableReason)
        assertTrue(state.status.isNotBlank())
    }

    @Test
    fun actionReturnsUnavailableAndDoesNotChangeNote() = runTest {
        val fakeAssistant = FakeUnavailableAssistant()
        val viewModel = AssistantViewModel(fakeAssistant)

        viewModel.onAction(AssistantUiAction.Summarize(NoteId("test-note-1")))
        advanceUntilIdle()

        val state = viewModel.state.value
        assertFalse(state.running)
        assertFalse(state.isAvailable)
        assertEquals("Inference models are not configured", state.unavailableReason)
        assertEquals("Inference models are not configured", state.status)

        assertEquals(0, fakeAssistant.applyCallsCount)
    }

    @Test
    fun rewriteActionReturnsUnavailableWithoutApply() = runTest {
        val fakeAssistant = FakeUnavailableAssistant()
        val viewModel = AssistantViewModel(fakeAssistant)

        viewModel.onAction(
            AssistantUiAction.Rewrite(
                noteId = NoteId("test-note-2"),
                contentItemId = ContentItemId("content-1"),
                style = AssistantRewriteStyle.CLEAR,
            )
        )
        advanceUntilIdle()

        val state = viewModel.state.value
        assertFalse(state.isAvailable)
        assertEquals("Inference models are not configured", state.unavailableReason)
        assertEquals(0, fakeAssistant.applyCallsCount)
    }
}
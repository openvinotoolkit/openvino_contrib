// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package com.openvino.notes.view.assistant

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.openvino.notes.assistant.api.AssistantRewriteStyle
import com.openvino.notes.assistant.api.NoteAssistant
import com.openvino.notes.assistant.api.SuggestionOutcome
import com.openvino.notes.notes.api.ContentItemId
import com.openvino.notes.notes.api.NoteId
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

data class AssistantUiState(val running: Boolean = false, val status: String = "Select an AI operation")

sealed interface AssistantUiAction {
    data class Summarize(val noteId: NoteId) : AssistantUiAction
    data class SuggestTextTags(val noteId: NoteId) : AssistantUiAction
    data class Rewrite(
        val noteId: NoteId,
        val contentItemId: ContentItemId,
        val style: AssistantRewriteStyle,
    ) : AssistantUiAction
}

class AssistantViewModel(private val assistant: NoteAssistant) : ViewModel() {
    private val mutableState = MutableStateFlow(AssistantUiState())
    val state: StateFlow<AssistantUiState> = mutableState.asStateFlow()

    fun onAction(action: AssistantUiAction) {
        viewModelScope.launch {
            mutableState.value = AssistantUiState(running = true, status = "Running")
            val outcome = when (action) {
                is AssistantUiAction.Summarize -> assistant.summarize(action.noteId)
                is AssistantUiAction.SuggestTextTags -> assistant.suggestTextTags(action.noteId)
                is AssistantUiAction.Rewrite -> assistant.rewrite(action.noteId, action.contentItemId, action.style)
            }
            mutableState.value = AssistantUiState(status = outcome.displayText())
        }
    }
}

private fun SuggestionOutcome.displayText(): String = when (this) {
    is SuggestionOutcome.Ready -> "Suggestion ready"
    SuggestionOutcome.NoteNotFound -> "Note not found"
    is SuggestionOutcome.InvalidTarget -> reason
    is SuggestionOutcome.Unavailable -> reason
    is SuggestionOutcome.Failed -> "Assistant failed: $code"
}

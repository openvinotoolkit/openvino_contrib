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

data class AssistantUiState(
    val running: Boolean = false,
    val status: String = "Assistant features are currently unavailable",
    val isAvailable: Boolean = false,
    val unavailableReason: String? = "OpenVINO model assets are not configured",
)

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
            mutableState.value = mutableState.value.copy(running = true, status = "Running")
            val outcome = when (action) {
                is AssistantUiAction.Summarize -> assistant.summarize(action.noteId)
                is AssistantUiAction.SuggestTextTags -> assistant.suggestTextTags(action.noteId)
                is AssistantUiAction.Rewrite -> assistant.rewrite(action.noteId, action.contentItemId, action.style)
            }
            mutableState.value = when (outcome) {
                is SuggestionOutcome.Unavailable -> AssistantUiState(
                    running = false,
                    status = outcome.reason,
                    isAvailable = false,
                    unavailableReason = outcome.reason,
                )
                is SuggestionOutcome.Ready -> AssistantUiState(
                    running = false,
                    status = "Suggestion ready",
                    isAvailable = true,
                    unavailableReason = null,
                )
                is SuggestionOutcome.InvalidTarget -> AssistantUiState(
                    running = false,
                    status = outcome.reason,
                    isAvailable = mutableState.value.isAvailable,
                )
                SuggestionOutcome.NoteNotFound -> AssistantUiState(
                    running = false,
                    status = "Note not found",
                    isAvailable = mutableState.value.isAvailable,
                )
                is SuggestionOutcome.Failed -> AssistantUiState(
                    running = false,
                    status = "Assistant failed: ${outcome.code}",
                    isAvailable = false,
                )
            }
        }
    }
}
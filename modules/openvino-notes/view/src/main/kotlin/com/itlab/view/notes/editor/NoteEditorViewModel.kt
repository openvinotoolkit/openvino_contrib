// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package com.itlab.view.notes.editor

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.itlab.notes.api.ContentItem
import com.itlab.notes.api.ContentItemId
import com.itlab.notes.api.FieldUpdate
import com.itlab.notes.api.NoteDraft
import com.itlab.notes.api.NoteId
import com.itlab.notes.api.NoteMutationOutcome
import com.itlab.notes.api.NotesService
import com.itlab.notes.api.UpdateNoteCommand
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

data class NoteEditorUiState(
    val noteId: NoteId? = null,
    val title: String = "",
    val text: String = "",
    val saving: Boolean = false,
    val error: String? = null,
)

sealed interface NoteEditorUiAction {
    data class Load(val noteId: NoteId) : NoteEditorUiAction
    data class ChangeTitle(val value: String) : NoteEditorUiAction
    data class ChangeText(val value: String) : NoteEditorUiAction
    data object Save : NoteEditorUiAction
}

class NoteEditorViewModel(private val notes: NotesService) : ViewModel() {
    private val mutableState = MutableStateFlow(NoteEditorUiState())
    val state: StateFlow<NoteEditorUiState> = mutableState.asStateFlow()

    fun onAction(action: NoteEditorUiAction) {
        when (action) {
            is NoteEditorUiAction.Load -> load(action.noteId)
            is NoteEditorUiAction.ChangeTitle -> mutableState.value = state.value.copy(title = action.value)
            is NoteEditorUiAction.ChangeText -> mutableState.value = state.value.copy(text = action.value)
            NoteEditorUiAction.Save -> save()
        }
    }

    private fun load(noteId: NoteId) {
        viewModelScope.launch {
            val note = notes.get(noteId)
            mutableState.value = if (note == null) {
                NoteEditorUiState(error = "Note not found")
            } else {
                NoteEditorUiState(
                    noteId = note.id,
                    title = note.title,
                    text = note.contentItems.filterIsInstance<ContentItem.Text>().joinToString("\n") { it.text },
                )
            }
        }
    }

    private fun save() {
        viewModelScope.launch {
            val snapshot = state.value
            mutableState.value = snapshot.copy(saving = true, error = null)
            val content = listOf(ContentItem.Text(ContentItemId("editor-body"), snapshot.text))
            val outcome = snapshot.noteId?.let { noteId ->
                notes.update(
                    UpdateNoteCommand(
                        noteId,
                        title = FieldUpdate.Set(snapshot.title),
                        contentItems = FieldUpdate.Set(content),
                    ),
                )
            } ?: notes.create(NoteDraft(snapshot.title, content))
            mutableState.value = when (outcome) {
                is NoteMutationOutcome.Saved -> snapshot.copy(noteId = outcome.note.id, saving = false)
                else -> snapshot.copy(saving = false, error = "Unable to save note")
            }
        }
    }
}

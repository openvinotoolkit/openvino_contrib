// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package com.openvino.notes.view.notes.editor

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.openvino.notes.notes.api.ContentItem
import com.openvino.notes.notes.api.ContentItemId
import com.openvino.notes.notes.api.FieldUpdate
import com.openvino.notes.notes.api.Note
import com.openvino.notes.notes.api.NoteDraft
import com.openvino.notes.notes.api.NoteId
import com.openvino.notes.notes.api.NoteMutationOutcome
import com.openvino.notes.notes.api.NotesService
import com.openvino.notes.notes.api.UpdateNoteCommand
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

data class NoteEditorUiState(
    val noteId: NoteId? = null,
    val title: String = "",
    val text: String = "",
    val contentItems: List<ContentItem> = emptyList(),
    val editableTextItemId: ContentItemId? = null,
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
                note.toEditorState()
            }
        }
    }

    private fun save() {
        viewModelScope.launch {
            val snapshot = state.value
            mutableState.value = snapshot.copy(saving = true, error = null)
            val content = snapshot.updatedContentItems()
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
                is NoteMutationOutcome.Saved -> outcome.note.toEditorState()
                else -> snapshot.copy(saving = false, error = "Unable to save note")
            }
        }
    }
}

private fun Note.toEditorState(): NoteEditorUiState {
    val editableText = contentItems.filterIsInstance<ContentItem.Text>().firstOrNull()
    return NoteEditorUiState(
        noteId = id,
        title = title,
        text = editableText?.text.orEmpty(),
        contentItems = contentItems,
        editableTextItemId = editableText?.id,
    )
}

private fun NoteEditorUiState.updatedContentItems(): List<ContentItem> {
    val targetId = editableTextItemId
    if (targetId != null) {
        return contentItems.map { item ->
            if (item is ContentItem.Text && item.id == targetId) item.copy(text = text) else item
        }
    }
    if (text.isEmpty()) return contentItems
    val usedIds = contentItems.map(ContentItem::id).toSet()
    val newId = generateSequence(0) { it + 1 }
        .map { suffix -> ContentItemId(if (suffix == 0) "editor-body" else "editor-body-$suffix") }
        .first { it !in usedIds }
    return contentItems + ContentItem.Text(newId, text)
}

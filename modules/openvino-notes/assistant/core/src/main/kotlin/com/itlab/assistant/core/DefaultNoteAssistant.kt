// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package com.itlab.assistant.core

import com.itlab.ai.image.api.ImageInput
import com.itlab.ai.image.api.ImageOutcome
import com.itlab.ai.image.api.ImageTagger
import com.itlab.ai.text.api.TextAssistant
import com.itlab.ai.text.api.TextOutcome
import com.itlab.ai.text.api.TextRequest
import com.itlab.assistant.api.ApplySuggestionOutcome
import com.itlab.assistant.api.AssistantSuggestion
import com.itlab.assistant.api.NoteAssistant
import com.itlab.assistant.api.SuggestionKind
import com.itlab.assistant.api.SuggestionOutcome
import com.itlab.notes.api.Note
import com.itlab.notes.api.NoteId
import com.itlab.notes.api.NoteMutationOutcome
import com.itlab.notes.api.NotesService
import com.itlab.notes.api.UpdateNoteCommand
import kotlinx.coroutines.flow.first

class DefaultNoteAssistant(
    private val notes: NotesService,
    private val text: TextAssistant,
    private val images: ImageTagger,
) : NoteAssistant {
    override suspend fun suggestTitle(noteId: NoteId): SuggestionOutcome = withNote(noteId) { note ->
        text.generate(TextRequest("Suggest a concise title. Return only the title.", note.body, 200)).toSuggestion(noteId, SuggestionKind.TITLE)
    }

    override suspend fun summarize(noteId: NoteId): SuggestionOutcome = withNote(noteId) { note ->
        text.generate(TextRequest("Summarize this note.", note.body)).toSuggestion(noteId, SuggestionKind.SUMMARY)
    }

    override suspend fun suggestImageTags(noteId: NoteId, bytes: ByteArray, mediaType: String): SuggestionOutcome = withNote(noteId) {
        when (val outcome = images.tag(ImageInput.Bytes(bytes, mediaType))) {
            is ImageOutcome.Tagged -> SuggestionOutcome.Ready(AssistantSuggestion(noteId, SuggestionKind.TAGS, outcome.tags.joinToString(", ") { it.label }))
            is ImageOutcome.NotConfigured -> SuggestionOutcome.Unavailable(outcome.reason)
            is ImageOutcome.Failed -> SuggestionOutcome.Failed(outcome.code)
        }
    }

    override suspend fun apply(suggestion: AssistantSuggestion): ApplySuggestionOutcome {
        val note = findNote(suggestion.noteId) ?: return ApplySuggestionOutcome.NoteNotFound
        val command = when (suggestion.kind) {
            SuggestionKind.TITLE -> UpdateNoteCommand(note.id, suggestion.value, note.body, note.folderId)
            SuggestionKind.SUMMARY, SuggestionKind.TAGS -> UpdateNoteCommand(note.id, note.title, note.body + "\n\n" + suggestion.value, note.folderId)
        }
        return when (notes.update(command)) {
            is NoteMutationOutcome.Saved -> ApplySuggestionOutcome.Applied
            NoteMutationOutcome.NotFound -> ApplySuggestionOutcome.NoteNotFound
            NoteMutationOutcome.SignedOut -> ApplySuggestionOutcome.Rejected("signed out")
            is NoteMutationOutcome.Invalid -> ApplySuggestionOutcome.Rejected("invalid suggestion")
            NoteMutationOutcome.Deleted -> ApplySuggestionOutcome.Rejected("unexpected delete result")
        }
    }

    private suspend fun withNote(noteId: NoteId, block: suspend (Note) -> SuggestionOutcome): SuggestionOutcome =
        findNote(noteId)?.let { block(it) } ?: SuggestionOutcome.NoteNotFound

    private suspend fun findNote(noteId: NoteId): Note? = notes.observeNotes().first().firstOrNull { it.id == noteId }
}

data class AssistantCoreComponent(val assistant: NoteAssistant) {
    companion object {
        fun create(notes: NotesService, text: TextAssistant, images: ImageTagger): AssistantCoreComponent =
            AssistantCoreComponent(DefaultNoteAssistant(notes, text, images))
    }
}

private fun TextOutcome.toSuggestion(noteId: NoteId, kind: SuggestionKind): SuggestionOutcome = when (this) {
    is TextOutcome.Generated -> SuggestionOutcome.Ready(AssistantSuggestion(noteId, kind, text.trim()))
    is TextOutcome.NotConfigured -> SuggestionOutcome.Unavailable(reason)
    is TextOutcome.Failed -> SuggestionOutcome.Failed(code)
}

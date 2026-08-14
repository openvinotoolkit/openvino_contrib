package com.itlab.assistant.api

import com.itlab.notes.api.NoteId

class FakeNoteAssistant : NoteAssistant {
    var outcome: SuggestionOutcome = SuggestionOutcome.Unavailable("fake not configured")
    override suspend fun suggestTitle(noteId: NoteId): SuggestionOutcome = outcome
    override suspend fun summarize(noteId: NoteId): SuggestionOutcome = outcome
    override suspend fun suggestImageTags(noteId: NoteId, bytes: ByteArray, mediaType: String): SuggestionOutcome = outcome
    override suspend fun apply(suggestion: AssistantSuggestion): ApplySuggestionOutcome = ApplySuggestionOutcome.Applied
}

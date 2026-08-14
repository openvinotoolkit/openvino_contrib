package com.itlab.assistant.api

import com.itlab.notes.api.NoteId

enum class SuggestionKind { TITLE, SUMMARY, TAGS }
data class AssistantSuggestion(val noteId: NoteId, val kind: SuggestionKind, val value: String)

sealed interface SuggestionOutcome {
    data class Ready(val suggestion: AssistantSuggestion) : SuggestionOutcome
    data object NoteNotFound : SuggestionOutcome
    data class Unavailable(val reason: String) : SuggestionOutcome
    data class Failed(val code: String) : SuggestionOutcome
}

sealed interface ApplySuggestionOutcome {
    data object Applied : ApplySuggestionOutcome
    data object NoteNotFound : ApplySuggestionOutcome
    data class Rejected(val reason: String) : ApplySuggestionOutcome
}

interface NoteAssistant {
    suspend fun suggestTitle(noteId: NoteId): SuggestionOutcome
    suspend fun summarize(noteId: NoteId): SuggestionOutcome
    suspend fun apply(suggestion: AssistantSuggestion): ApplySuggestionOutcome
}

package com.itlab.notes.ui

import com.itlab.notes.ui.notes.NoteItemUi
import kotlinx.coroutines.CancellationException

internal enum class AiSuggestion {
    Summary,
    Tags,
    Rewrite,
}

internal suspend fun generateAiSuggestion(
    suggestion: AiSuggestion,
    savedNote: NoteItemUi,
    useCases: NotesUseCases,
    currentEditorNote: () -> NoteItemUi,
    ensureCurrentEditorSnapshot: () -> Unit,
): Result<NoteItemUi> =
    when (suggestion) {
        AiSuggestion.Summary ->
            useCases
                .suggestSummaryUseCase(savedNote.id)
                .mapCatching { summary ->
                    currentEditorNote()
                    useCases.applySummaryUseCase(savedNote.id, summary).getOrThrow()
                    currentEditorNote().copy(summary = summary)
                }
        AiSuggestion.Tags ->
            useCases
                .suggestTagsUseCase(savedNote.id)
                .mapCatching { tags ->
                    currentEditorNote()
                    useCases.applyTagsUseCase(savedNote.id, tags).getOrThrow()
                    currentEditorNote().copy(tags = tags)
                }
        AiSuggestion.Rewrite ->
            useCases
                .rewriteNoteUseCase(savedNote.id)
                .mapCatching { rewrittenContent ->
                    ensureCurrentEditorSnapshot()
                    useCases.applyRewriteUseCase(savedNote.id, rewrittenContent).getOrThrow()
                    savedNote.copy(content = rewrittenContent)
                }
    }

internal fun NotesUiState.isCurrentEditorNote(noteId: String): Boolean =
    (screen as? NotesUiScreen.NoteEditor)?.note?.id == noteId

internal fun NotesUiState.requireCurrentEditorNote(noteId: String): NoteItemUi {
    val currentNote = (screen as? NotesUiScreen.NoteEditor)?.note
    if (currentNote?.id != noteId) {
        throw CancellationException("Editor changed before AI generation completed.")
    }
    return currentNote
}

internal fun NotesUiState.requireCurrentEditorSnapshot(note: NoteItemUi) {
    val currentNote = (screen as? NotesUiScreen.NoteEditor)?.note
    if (currentNote?.id != note.id) {
        throw CancellationException("Editor changed before AI generation completed.")
    }
    if (currentNote != note) {
        throw CancellationException("Editor content changed before AI rewrite completed.")
    }
}

internal fun AiSuggestion.startState(state: AiUiState): AiUiState =
    when (this) {
        AiSuggestion.Summary -> state.copy(isGeneratingSummary = true, errorMessage = null)
        AiSuggestion.Tags -> state.copy(isGeneratingTags = true, errorMessage = null)
        AiSuggestion.Rewrite -> state.copy(isRewriting = true, errorMessage = null)
    }

internal fun AiSuggestion.successState(state: AiUiState): AiUiState =
    when (this) {
        AiSuggestion.Summary -> state.copy(isGeneratingSummary = false, errorMessage = null)
        AiSuggestion.Tags -> state.copy(isGeneratingTags = false, errorMessage = null)
        AiSuggestion.Rewrite -> state.copy(isRewriting = false, errorMessage = null)
    }

internal fun AiSuggestion.errorState(
    state: AiUiState,
    error: Throwable,
): AiUiState =
    when (this) {
        AiSuggestion.Summary ->
            state.copy(
                isGeneratingSummary = false,
                errorMessage = error.userMessage("Unable to generate summary"),
            )
        AiSuggestion.Tags ->
            state.copy(
                isGeneratingTags = false,
                errorMessage = error.userMessage("Unable to suggest AI tags"),
            )
        AiSuggestion.Rewrite ->
            state.copy(
                isRewriting = false,
                errorMessage = error.userMessage("Unable to rewrite note"),
            )
    }

internal fun Throwable.userMessage(fallback: String): String = message ?: fallback

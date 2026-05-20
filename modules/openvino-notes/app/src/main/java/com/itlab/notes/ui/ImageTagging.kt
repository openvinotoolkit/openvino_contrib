package com.itlab.notes.ui

import com.itlab.notes.ui.notes.NoteItemUi

internal suspend fun generateImageTags(
    savedNote: NoteItemUi,
    useCases: NotesUseCases,
    currentEditorNote: () -> NoteItemUi,
): Result<NoteItemUi> =
    useCases
        .suggestImageTagsUseCase(savedNote.id)
        .mapCatching { tags ->
            currentEditorNote()
            useCases.applyTagsUseCase(savedNote.id, tags).getOrThrow()
            currentEditorNote().copy(tags = tags)
        }

internal fun ImageTaggingUiState.startTagging(): ImageTaggingUiState = copy(isTagging = true, errorMessage = null)

internal fun ImageTaggingUiState.finishTagging(): ImageTaggingUiState = copy(isTagging = false, errorMessage = null)

internal fun ImageTaggingUiState.failTagging(error: Throwable): ImageTaggingUiState =
    copy(
        isTagging = false,
        errorMessage = error.userMessage("Unable to tag images"),
    )

package com.itlab.notes.ui

import com.itlab.notes.ui.notes.DirectoryItemUi
import com.itlab.notes.ui.notes.NoteItemUi
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch

internal class NotesAiController(
    private val scope: CoroutineScope,
    private val useCases: NotesUseCases,
    private val getUiState: () -> NotesUiState,
    private val setUiState: (NotesUiState) -> Unit,
    private val upsertEditorNote: suspend (NoteItemUi, DirectoryItemUi) -> Result<NoteItemUi>,
    private val updateEditorNote: (NoteItemUi) -> Unit,
) {
    private var aiJob: Job? = null
    private var imageTaggingJob: Job? = null
    private var warmUpJob: Job? = null
    private var warmUpStarted = false
    private var ready = false

    fun suggestAi(
        note: NoteItemUi,
        suggestion: AiSuggestion,
    ) {
        if (aiJob?.isActive == true || imageTaggingJob?.isActive == true) return
        if (!getUiState().aiState.canGenerate) {
            warmUp()
            return
        }

        val editor = getUiState().screen as? NotesUiScreen.NoteEditor ?: return
        val job =
            scope.launch {
                updateAiState { suggestion.startState(it) }
                val savedNote =
                    upsertEditorNote(note, editor.directory)
                        .getOrElse { error ->
                            updateAiState { suggestion.errorState(it, error) }
                            return@launch
                        }
                updateEditorNote(savedNote)

                val generated =
                    generateAiSuggestion(
                        suggestion = suggestion,
                        savedNote = savedNote,
                        useCases = useCases,
                        currentEditorNote = { getUiState().requireCurrentEditorNote(savedNote.id) },
                        ensureCurrentEditorSnapshot = { getUiState().requireCurrentEditorSnapshot(savedNote) },
                    )

                generated
                    .onSuccess { updatedNote ->
                        if (getUiState().isCurrentEditorNote(savedNote.id)) {
                            updateEditorNote(updatedNote)
                            updateAiState { suggestion.successState(it) }
                        }
                    }.onFailure { error ->
                        if (getUiState().isCurrentEditorNote(savedNote.id)) {
                            updateAiState {
                                if (error is CancellationException) {
                                    suggestion.successState(it)
                                } else {
                                    suggestion.errorState(it, error)
                                }
                            }
                        }
                    }
            }
        aiJob = job
        job.invokeOnCompletion {
            if (aiJob === job) {
                aiJob = null
            }
        }
    }

    fun suggestImageTags(note: NoteItemUi) {
        if (imageTaggingJob?.isActive == true || aiJob?.isActive == true) return

        val editor = getUiState().screen as? NotesUiScreen.NoteEditor ?: return
        val job =
            scope.launch {
                updateImageTaggingState { it.startTagging() }
                val savedNote =
                    upsertEditorNote(note, editor.directory)
                        .getOrElse { error ->
                            updateImageTaggingState { it.failTagging(error) }
                            return@launch
                        }
                updateEditorNote(savedNote)

                val generated =
                    generateImageTags(
                        savedNote = savedNote,
                        useCases = useCases,
                        currentEditorNote = { getUiState().requireCurrentEditorNote(savedNote.id) },
                    )

                generated
                    .onSuccess { updatedNote ->
                        if (getUiState().isCurrentEditorNote(savedNote.id)) {
                            updateEditorNote(updatedNote)
                            updateImageTaggingState { it.finishTagging() }
                        }
                    }.onFailure { error ->
                        if (getUiState().isCurrentEditorNote(savedNote.id)) {
                            updateImageTaggingState {
                                if (error is CancellationException) {
                                    it.finishTagging()
                                } else {
                                    it.failTagging(error)
                                }
                            }
                        }
                    }
            }
        imageTaggingJob = job
        job.invokeOnCompletion {
            if (imageTaggingJob === job) {
                imageTaggingJob = null
            }
        }
    }

    fun cancelGeneration() {
        aiJob?.cancel()
        aiJob = null
        imageTaggingJob?.cancel()
        imageTaggingJob = null
        updateAiState { freshAiState() }
        updateImageTaggingState { ImageTaggingUiState() }
    }

    fun warmUp() {
        if (ready) {
            updateAiState { it.copy(isWarmingUp = false, isReady = true, errorMessage = null) }
            return
        }
        if (warmUpStarted || warmUpJob?.isActive == true) {
            updateAiState { it.copy(isWarmingUp = true, isReady = false, errorMessage = null) }
            return
        }

        warmUpStarted = true
        updateAiState { it.copy(isWarmingUp = true, isReady = false, errorMessage = null) }
        val job =
            scope.launch {
                val result = useCases.warmUpNoteAiUseCase()
                if (result.isSuccess) {
                    ready = true
                    updateAiState { it.copy(isWarmingUp = false, isReady = true, errorMessage = null) }
                } else {
                    if (result.exceptionOrNull() is CancellationException) {
                        return@launch
                    }
                    ready = false
                    warmUpStarted = false
                    updateAiState {
                        it.copy(
                            isWarmingUp = false,
                            isReady = false,
                            errorMessage =
                                result
                                    .exceptionOrNull()
                                    ?.userMessage("Unable to prepare AI model")
                                    ?: "Unable to prepare AI model",
                        )
                    }
                }
            }
        warmUpJob = job
        job.invokeOnCompletion { error ->
            if (warmUpJob === job) {
                warmUpJob = null
            }
            if (error is CancellationException && !ready) {
                warmUpStarted = false
            }
        }
    }

    fun freshAiState(): AiUiState =
        AiUiState(
            isWarmingUp = warmUpJob?.isActive == true,
            isReady = ready,
        )

    fun cancelAll() {
        aiJob?.cancel()
        imageTaggingJob?.cancel()
        warmUpJob?.cancel()
    }

    private fun updateAiState(update: (AiUiState) -> AiUiState) {
        val state = getUiState()
        setUiState(state.copy(aiState = update(state.aiState)))
    }

    private fun updateImageTaggingState(update: (ImageTaggingUiState) -> ImageTaggingUiState) {
        val state = getUiState()
        setUiState(state.copy(imageTaggingState = update(state.imageTaggingState)))
    }
}

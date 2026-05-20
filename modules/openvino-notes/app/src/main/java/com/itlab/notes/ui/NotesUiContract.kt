package com.itlab.notes.ui

import com.itlab.notes.ui.notes.DirectoryItemUi
import com.itlab.notes.ui.notes.NoteItemUi

/**
 * UI contract for the Notes feature.
 * Keeps state & events in one place so screens stay "dumb" (render-only).
 */
sealed interface NotesUiScreen {
    data object Directories : NotesUiScreen

    data class DirectoryNotes(
        val directory: DirectoryItemUi,
    ) : NotesUiScreen

    data class NoteEditor(
        val directory: DirectoryItemUi,
        val note: NoteItemUi,
        val cloudSyncStatus: EditorCloudSyncStatus = EditorCloudSyncStatus.Idle,
    ) : NotesUiScreen
}

/** Cloud upload state shown in the editor top bar. */
enum class EditorCloudSyncStatus {
    Idle,
    Uploading,
    Error,
}

data class NotesUiState(
    val screen: NotesUiScreen = NotesUiScreen.Directories,
    val directories: List<DirectoryItemUi> = emptyList(),
    val notes: List<NoteItemUi> = emptyList(),
    val notesSearchQuery: String = "",
    val directorySearchQuery: String = "",
    /** Note ids currently uploading to cloud (visible in list + editor). */
    val noteIdsUploading: Set<String> = emptySet(),
    /** Pull-to-refresh / download from cloud in progress. */
    val isCloudDownloadActive: Boolean = false,
)

sealed interface NotesUiEvent {
    data class OpenDirectory(
        val directory: DirectoryItemUi,
    ) : NotesUiEvent

    data object BackToDirectories : NotesUiEvent

    data class OpenNote(
        val note: NoteItemUi,
    ) : NotesUiEvent

    data object CreateNote : NotesUiEvent

    data class CreateDirectory(
        val name: String,
    ) : NotesUiEvent

    data class DeleteDirectory(
        val directoryId: String,
    ) : NotesUiEvent

    data object BackToDirectoryNotes : NotesUiEvent

    /** Saves pending editor changes (if any), then returns to the notes list. */
    data class LeaveEditor(
        val note: NoteItemUi,
    ) : NotesUiEvent

    /** Persists editor changes without leaving the editor screen. */
    data class PersistNote(
        val note: NoteItemUi,
    ) : NotesUiEvent

    data class DeleteNote(
        val noteId: String,
    ) : NotesUiEvent

    data class RenameDirectory(
        val directoryId: String,
        val newName: String,
    ) : NotesUiEvent

    data class MoveNoteToDirectory(
        val noteId: String,
        val targetDirectoryId: String,
    ) : NotesUiEvent

    data class NotesSearchQueryChanged(
        val query: String,
    ) : NotesUiEvent

    data class DirectorySearchQueryChanged(
        val query: String,
    ) : NotesUiEvent

    data class ToggleNoteFavorite(
        val noteId: String,
    ) : NotesUiEvent

    data object SyncCloud : NotesUiEvent
}

interface NotesViewModelContract {
    val uiState: NotesUiState

    fun onEvent(event: NotesUiEvent)
}

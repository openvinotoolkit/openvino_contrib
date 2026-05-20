package com.itlab.notes.ui

import com.itlab.domain.model.Note
import com.itlab.domain.model.NoteFolder
import com.itlab.notes.ui.notes.ALL_DIRECTORY_ID
import com.itlab.notes.ui.notes.DirectoryItemUi
import com.itlab.notes.ui.notes.FAVORITES_DIRECTORY_ID
import com.itlab.notes.ui.notes.RECENT_DIRECTORY_ID
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map

internal fun notesFlowForDirectory(
    useCases: NotesUseCases,
    folders: List<NoteFolder>,
    directory: DirectoryItemUi,
    searchQuery: String,
): Flow<List<Note>> {
    val normalizedQuery = searchQuery.trim()
    return if (normalizedQuery.isBlank()) {
        notesFlowForBlankQuery(useCases, folders, directory)
    } else {
        notesFlowForSearchQuery(useCases, folders, directory, normalizedQuery)
    }
}

internal fun buildDirectoryItems(
    folders: List<NoteFolder>,
    notes: List<Note>,
): List<DirectoryItemUi> {
    val activeNotes = notesInActiveFolders(folders, notes)
    val countsByFolderId = activeNotes.groupingBy { it.folderId }.eachCount()
    val allNotesDir = DirectoryItemUi(id = ALL_DIRECTORY_ID, name = "All Notes", noteCount = activeNotes.size)
    val favoritesDir =
        DirectoryItemUi(
            id = FAVORITES_DIRECTORY_ID,
            name = "Favorites",
            noteCount = activeNotes.count { it.isFavorite },
        )

    return listOf(allNotesDir, favoritesDir) +
        folders.map { folder ->
            folder.toUi(noteCount = countsByFolderId[folder.id] ?: 0)
        }
}

internal fun NotesUiScreen.withUpdatedDirectoryCount(directories: List<DirectoryItemUi>): NotesUiScreen {
    val opened = this as? NotesUiScreen.DirectoryNotes ?: return this
    val updatedDir = directories.firstOrNull { it.id == opened.directory.id } ?: return this
    return if (updatedDir.noteCount == opened.directory.noteCount) {
        this
    } else {
        NotesUiScreen.DirectoryNotes(directory = updatedDir)
    }
}

private fun notesFlowForBlankQuery(
    useCases: NotesUseCases,
    folders: List<NoteFolder>,
    directory: DirectoryItemUi,
): Flow<List<Note>> =
    when (directory.id) {
        ALL_DIRECTORY_ID ->
            useCases.observeNotesUseCase().map { notesInActiveFolders(folders, it) }
        FAVORITES_DIRECTORY_ID ->
            useCases.getAllFavoritesUseCase().map { notesInActiveFolders(folders, it) }
        RECENT_DIRECTORY_ID ->
            useCases.observeNotesUseCase().map { notes ->
                notesInActiveFolders(folders, notes).sortedByDescending { it.updatedAt }
            }
        else -> useCases.observeNotesByFolderUseCase(directory.id)
    }

private fun notesFlowForSearchQuery(
    useCases: NotesUseCases,
    folders: List<NoteFolder>,
    directory: DirectoryItemUi,
    searchQuery: String,
): Flow<List<Note>> {
    val searchFlow =
        useCases.searchNotesUseCase(
            query = searchQuery,
            folderId = directory.folderIdForSearch(),
        )
    return when (directory.id) {
        FAVORITES_DIRECTORY_ID ->
            searchFlow.map { notes ->
                notesInActiveFolders(folders, notes).filter { it.isFavorite }
            }
        ALL_DIRECTORY_ID -> searchFlow.map { notesInActiveFolders(folders, it) }
        RECENT_DIRECTORY_ID ->
            searchFlow.map { notes ->
                notesInActiveFolders(folders, notes).sortedByDescending { it.updatedAt }
            }
        else -> searchFlow
    }
}

/** Notes whose folder was deleted stay in DB until sync; hide them from All/Recent. */
private fun notesInActiveFolders(
    folders: List<NoteFolder>,
    notes: List<Note>,
): List<Note> {
    val activeFolderIds = folders.map { it.id }.toSet()
    return notes.filter { note ->
        val folderId = note.folderId ?: return@filter true
        folderId in activeFolderIds
    }
}

package com.itlab.domain.usecase.noteusecase

import com.itlab.domain.repository.NoteFolderRepository
import com.itlab.domain.repository.NotesRepository
import com.itlab.domain.usecase.requireNotBlank
import kotlin.time.Clock

class MoveNoteToFolderUseCase(
    private val notesRepo: NotesRepository,
    private val folderRepo: NoteFolderRepository,
    private val getUserIdUseCase: GetUserIdUseCase,
) {
    suspend operator fun invoke(
        folderId: String,
        noteId: String,
    ): Result<Unit> =
        runCatching {
            val userId = getUserIdUseCase()

            if (userId == null) {
                return Result.failure(IllegalStateException("User must be authenticated"))
            }

            requireNotBlank(noteId, "Note id")
            requireNotBlank(folderId, "Folder id")
            requireNotNull(folderRepo.getFolderById(folderId, userId)) { "Folder not found: $folderId" }
            val note =
                notesRepo.getNoteById(noteId, userId) ?: throw IllegalArgumentException("Note not found: $noteId")
            val updated = note.copy(folderId = folderId, updatedAt = Clock.System.now())
            notesRepo.updateNote(updated)
        }
}

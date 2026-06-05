package com.itlab.domain.usecase.folderusecase

import com.itlab.domain.repository.NoteFolderRepository
import com.itlab.domain.repository.NotesRepository
import com.itlab.domain.usecase.noteusecase.GetUserIdUseCase
import com.itlab.domain.usecase.requireNotBlank
import kotlinx.coroutines.flow.first

class DeleteFolderUseCase(
    private val repo: NoteFolderRepository,
    private val notesRepository: NotesRepository,
    private val getUserIdUseCase: GetUserIdUseCase,
) {
    suspend operator fun invoke(id: String): Result<Unit> =
        runCatching {
            requireNotBlank(id, "Folder id")
            require(id != "all") { "System folder 'all' cannot be deleted" }

            val userId = getUserIdUseCase()
            if (userId == null) {
                return Result.failure(IllegalStateException("User must be authenticated to delete folders"))
            }

            notesRepository
                .observeNotesByFolder(id, userId)
                .first()
                .forEach { note ->
                    notesRepository.deleteNote(note.id, userId)
                }
            repo.deleteFolder(id, userId)
        }
}

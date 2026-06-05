package com.itlab.domain.usecase.noteusecase

import com.itlab.domain.repository.NotesRepository

class SwitchFavoriteUseCase(
    private val repo: NotesRepository,
    private val getUserIdUseCase: GetUserIdUseCase,
) {
    suspend operator fun invoke(noteId: String): Result<Unit> =
        runCatching {
            val userId = getUserIdUseCase()

            if (userId == null) {
                return Result.failure(IllegalStateException("User must be authenticated"))
            }

            val note =
                repo.getNoteById(noteId, userId)
                    ?: throw IllegalArgumentException("Note not found")

            val updatedNote =
                note.copy(
                    isFavorite = !note.isFavorite,
                )

            repo.updateNote(updatedNote)
        }
}

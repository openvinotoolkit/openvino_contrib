package com.itlab.domain.usecase.noteusecase

import com.itlab.domain.repository.NotesRepository

class DeleteNoteUseCase(
    private val repo: NotesRepository,
    private val getUserIdUseCase: GetUserIdUseCase,
) {
    suspend operator fun invoke(noteId: String): Result<Unit> =
        runCatching {
            val userId =
                checkNotNull(getUserIdUseCase()) {
                    "User must be authenticated to delete notes"
                }

            repo.deleteNote(noteId, userId)
        }
}

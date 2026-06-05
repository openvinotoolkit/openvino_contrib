package com.itlab.domain.usecase.noteusecase

import com.itlab.domain.model.Note
import com.itlab.domain.repository.NotesRepository

class GetNoteUseCase(
    private val repo: NotesRepository,
    private val getUserIdUseCase: GetUserIdUseCase,
) {
    suspend operator fun invoke(id: String): Note? {
        val userId = getUserIdUseCase() ?: return null

        return repo.getNoteById(id, userId)
    }
}

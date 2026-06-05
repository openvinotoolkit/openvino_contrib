package com.itlab.domain.usecase.noteusecase

import com.itlab.domain.model.Note
import com.itlab.domain.repository.NotesRepository
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flowOf
import kotlinx.coroutines.flow.map

class GetAllFavoritesUseCase(
    private val repo: NotesRepository,
    private val getUserIdUseCase: GetUserIdUseCase,
) {
    operator fun invoke(): Flow<List<Note>> {
        val userId = getUserIdUseCase()

        if (userId == null) {
            return flowOf(emptyList())
        }
        return repo.observeNotes(userId).map { notes ->
            notes.filter { it.isFavorite }
        }
    }
}

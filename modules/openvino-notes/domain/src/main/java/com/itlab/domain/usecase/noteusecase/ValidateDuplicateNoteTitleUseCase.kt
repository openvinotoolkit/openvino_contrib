package com.itlab.domain.usecase.noteusecase

import com.itlab.domain.repository.NotesRepository
import kotlinx.coroutines.flow.first

class ValidateDuplicateNoteTitleUseCase(
    private val repo: NotesRepository,
    private val getUserIdUseCase: GetUserIdUseCase,
) {
    /** @return true when another note in the same folder already has this title. */
    suspend operator fun invoke(
        title: String,
        folderId: String?,
        excludeNoteId: String,
    ): Boolean {
        val userId = getUserIdUseCase()
        val normalizedTitle = title.trim()

        if (userId == null || normalizedTitle.isEmpty()) {
            return false
        }

        return repo.observeNotes(userId).first().any { existing ->
            existing.id != excludeNoteId &&
                existing.folderId == folderId &&
                existing.title.trim().equals(normalizedTitle, ignoreCase = true)
        }
    }
}

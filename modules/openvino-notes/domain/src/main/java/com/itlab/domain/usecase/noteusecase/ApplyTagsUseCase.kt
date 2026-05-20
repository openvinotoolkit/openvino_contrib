package com.itlab.domain.usecase.noteusecase

import com.itlab.domain.repository.NotesRepository
import com.itlab.domain.usecase.requireNotBlank
import kotlin.time.Clock

class ApplyTagsUseCase(
    private val repo: NotesRepository,
    private val getUserIdUseCase: GetUserIdUseCase,
) {
    suspend operator fun invoke(
        noteId: String,
        newTags: Set<String>,
    ): Result<Unit> =
        runCatching {
            val userId = getUserIdUseCase()

            if (userId == null) {
                return Result.failure(IllegalStateException("User must be authenticated"))
            }

            requireNotBlank(noteId, "Note id")
            val note =
                repo.getNoteById(noteId, userId)
                    ?: throw IllegalArgumentException("Note not found")
            val normalizedTags = newTags.map { it.trim() }.filter { it.isNotBlank() }.toSet()

            val updated =
                note.copy(
                    tags = normalizedTags,
                    updatedAt =
                        Clock.System
                            .now(),
                )

            repo.updateNote(updated)
        }
}

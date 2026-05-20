package com.itlab.domain.usecase.noteusecase

import com.itlab.domain.repository.NotesRepository
import com.itlab.domain.usecase.requireNotBlank
import kotlin.time.Clock

class DeleteTagUseCase(
    private val repo: NotesRepository,
    private val getUserIdUseCase: GetUserIdUseCase,
) {
    suspend operator fun invoke(
        noteId: String,
        tagToDel: String,
    ): Result<Unit> =
        runCatching {
            val userId = getUserIdUseCase()

            if (userId == null) {
                return Result.failure(IllegalStateException("User must be authenticated"))
            }

            requireNotBlank(noteId, "Note id")
            val normalizedTag = tagToDel.trim()
            requireNotBlank(normalizedTag, "Tag")

            val note =
                repo.getNoteById(noteId, userId)
                    ?: throw IllegalArgumentException("Note not found: $noteId")

            val updated =
                note.copy(
                    tags = note.tags - normalizedTag,
                    updatedAt = Clock.System.now(),
                )

            repo.updateNote(updated)
        }
}

package com.itlab.domain.usecase.contentusecase

import com.itlab.domain.model.ContentItem
import com.itlab.domain.repository.NotesRepository
import com.itlab.domain.usecase.noteusecase.GetUserIdUseCase
import com.itlab.domain.usecase.requireNotBlank
import kotlin.time.Clock

class AddContentItemUseCase(
    private val notesRepository: NotesRepository,
    private val getUserIdUseCase: GetUserIdUseCase,
) {
    suspend operator fun invoke(
        noteId: String,
        item: ContentItem,
    ): Result<Unit> =
        runCatching {
            val userId = getUserIdUseCase()

            if (userId == null) {
                return Result.failure(IllegalStateException("User must be authenticated"))
            }

            requireNotBlank(noteId, "Note id")
            requireNotBlank(item.id, "ContentItem id")

            val note =
                notesRepository.getNoteById(noteId, userId)
                    ?: throw IllegalArgumentException("Note not found: $noteId")

            val updated =
                note.copy(
                    contentItems =
                        note.contentItems.also {
                            require(it.none { existing -> existing.id == item.id }) {
                                "Content item with id '${item.id}' already exists in note '$noteId'"
                            }
                        } + item,
                    updatedAt = Clock.System.now(),
                )

            notesRepository.updateNote(updated)
        }
}

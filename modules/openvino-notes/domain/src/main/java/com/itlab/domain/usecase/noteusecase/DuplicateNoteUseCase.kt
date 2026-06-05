package com.itlab.domain.usecase.noteusecase

import com.itlab.domain.model.ContentItem
import com.itlab.domain.repository.NotesRepository
import java.util.UUID
import kotlin.time.Clock

class DuplicateNoteUseCase(
    private val repo: NotesRepository,
    private val getUserIdUseCase: GetUserIdUseCase,
) {
    suspend operator fun invoke(noteId: String): Result<String> =
        runCatching {
            val userId = getUserIdUseCase()

            if (userId == null) {
                return Result.failure(IllegalStateException("User must be authenticated"))
            }

            val note =
                repo.getNoteById(noteId, userId)
                    ?: throw IllegalArgumentException("Note not found: $noteId")

            val now = Clock.System.now()
            val duplicated =
                note.copy(
                    id = UUID.randomUUID().toString(),
                    title = if (note.title.isBlank()) "Copy" else "${note.title} Copy",
                    createdAt = now,
                    updatedAt = now,
                    contentItems =
                        note.contentItems.map { item ->
                            when (item) {
                                is ContentItem.Text -> item.copy(id = UUID.randomUUID().toString())
                                is ContentItem.Image -> item.copy(id = UUID.randomUUID().toString())
                                is ContentItem.File -> item.copy(id = UUID.randomUUID().toString())
                                is ContentItem.Link -> item.copy(id = UUID.randomUUID().toString())
                            }
                        },
                )
            repo.createNote(duplicated)
        }
}

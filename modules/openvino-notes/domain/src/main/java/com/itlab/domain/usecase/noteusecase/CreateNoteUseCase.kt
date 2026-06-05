package com.itlab.domain.usecase.noteusecase

import com.itlab.domain.model.Note
import com.itlab.domain.repository.NotesRepository
import kotlinx.coroutines.flow.first
import java.util.UUID
import kotlin.time.Clock

class CreateNoteUseCase(
    private val repo: NotesRepository,
    private val getUserIdUseCase: GetUserIdUseCase,
) {
    suspend operator fun invoke(note: Note): Result<String> =
        runCatching {
            val userId = getUserIdUseCase()
            if (userId == null) {
                return Result.failure(IllegalStateException("User must be authenticated"))
            }

            val normalizedTitle = note.title.trim()
            val hasDuplicateTitle =
                repo.observeNotes(userId).first().any { existing ->
                    existing.folderId == note.folderId &&
                        existing.title.trim().equals(normalizedTitle, ignoreCase = true)
                }
            require(!hasDuplicateTitle) { "Note with title '$normalizedTitle' already exists in this folder" }
            val now = Clock.System.now()

            val noteId = note.id.takeIf { it.isNotBlank() } ?: UUID.randomUUID().toString()
            val noteToSave =
                note.copy(
                    id = noteId,
                    userId = userId,
                    createdAt = now,
                    updatedAt = now,
                )
            repo.createNote(noteToSave)
        }
}

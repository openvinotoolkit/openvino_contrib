package com.itlab.domain.usecase.noteusecase

import com.itlab.domain.model.ContentItem
import com.itlab.domain.repository.NotesRepository
import kotlin.time.Clock

class ApplyRewriteUseCase(
    private val repo: NotesRepository,
    private val getUserIdUseCase: GetUserIdUseCase,
) {
    suspend operator fun invoke(
        noteId: String,
        rewrittenText: String,
    ): Result<Unit> =
        runCatching {
            val userId =
                getUserIdUseCase()
                    ?: return Result.failure(IllegalStateException("User must be authenticated"))
            val note =
                repo.getNoteById(noteId, userId)
                    ?: throw IllegalArgumentException("Note not found")
            val normalizedText = rewrittenText.trim()
            val nonTextContent = note.contentItems.filterNot { it is ContentItem.Text }
            val updatedText =
                normalizedText
                    .takeIf { it.isNotBlank() }
                    ?.let { ContentItem.Text(text = it) }

            repo.updateNote(
                note.copy(
                    contentItems = if (updatedText != null) nonTextContent + updatedText else nonTextContent,
                    updatedAt = Clock.System.now(),
                ),
            )
        }
}

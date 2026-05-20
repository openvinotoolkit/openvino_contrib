package com.itlab.domain.usecase.aiusecase

import com.itlab.domain.ai.NoteAiService
import com.itlab.domain.ai.RewriteStyle
import com.itlab.domain.model.ContentItem
import com.itlab.domain.model.Note
import com.itlab.domain.repository.NotesRepository
import com.itlab.domain.usecase.noteusecase.GetUserIdUseCase

class RewriteNoteUseCase(
    private val ai: NoteAiService,
    private val repo: NotesRepository,
    private val getUserIdUseCase: GetUserIdUseCase,
) {
    private fun extractText(note: Note): String =
        note.contentItems
            .filterIsInstance<ContentItem.Text>()
            .joinToString("\n") { it.text }

    suspend operator fun invoke(
        noteId: String,
        style: RewriteStyle = RewriteStyle.CLEANUP,
        maxInputTokens: Int = 768,
        maxNewTokens: Int = 192,
    ): Result<String> =
        runCatching {
            val userId =
                getUserIdUseCase()
                    ?: return Result.failure(IllegalStateException("User must be authenticated"))
            val note =
                repo.getNoteById(noteId, userId)
                    ?: throw IllegalArgumentException("Note not found: $noteId")

            ai.rewrite(
                text = extractText(note),
                style = style,
                maxInputTokens = maxInputTokens,
                maxNewTokens = maxNewTokens,
            )
        }
}

package com.itlab.domain.usecase.aiusecase

import com.itlab.domain.ai.NoteAiService
import com.itlab.domain.model.ContentItem
import com.itlab.domain.model.Note
import com.itlab.domain.repository.NotesRepository
import com.itlab.domain.usecase.noteusecase.GetUserIdUseCase

class SuggestTagsUseCase(
    private val ai: NoteAiService,
    private val repo: NotesRepository,
    private val getUserIdUseCase: GetUserIdUseCase,
) {
    private fun extractText(note: Note): String =
        note.contentItems
            .filterIsInstance<ContentItem.Text>()
            .joinToString("\n") { it.text }

    private fun extractImages(note: Note): List<String> =
        note.contentItems
            .filterIsInstance<ContentItem.Image>()
            .mapNotNull { image ->
                image.source.localPath ?: image.source.remoteUrl
            }

    suspend operator fun invoke(noteId: String): Result<Set<String>> =
        runCatching {
            val userId = getUserIdUseCase()

            if (userId == null) {
                return Result.failure(IllegalStateException("User must be authenticated"))
            }

            val note =
                repo.getNoteById(noteId, userId)
                    ?: throw IllegalArgumentException("Note not found: $noteId")

            val text = extractText(note)
            val imageUrls = extractImages(note)

            ai.tagTXT(text) + ai.tagIMGs(imageUrls)
        }
}

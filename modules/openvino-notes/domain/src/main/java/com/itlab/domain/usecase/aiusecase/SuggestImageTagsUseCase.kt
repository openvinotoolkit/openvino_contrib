package com.itlab.domain.usecase.aiusecase

import com.itlab.domain.ai.NoteAiService
import com.itlab.domain.model.ContentItem
import com.itlab.domain.model.Note
import com.itlab.domain.repository.NotesRepository
import com.itlab.domain.usecase.noteusecase.GetUserIdUseCase

class SuggestImageTagsUseCase(
    private val ai: NoteAiService,
    private val repo: NotesRepository,
    private val getUserIdUseCase: GetUserIdUseCase,
) {
    private fun extractImages(note: Note): List<String> =
        note.contentItems
            .filterIsInstance<ContentItem.Image>()
            .mapNotNull { image ->
                image.source.localPath ?: image.source.remoteUrl
            }

    suspend operator fun invoke(
        noteId: String,
        maxTags: Int = 4,
    ): Result<Set<String>> =
        runCatching {
            val userId =
                getUserIdUseCase()
                    ?: return Result.failure(IllegalStateException("User must be authenticated"))
            val note =
                repo.getNoteById(noteId, userId)
                    ?: throw IllegalArgumentException("Note not found: $noteId")

            val imageTags = ai.tagIMGs(extractImages(note))
            imageTags
                .take(maxTags.coerceAtLeast(0))
                .toSet()
        }
}

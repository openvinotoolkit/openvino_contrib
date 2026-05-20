package com.itlab.domain.usecase.noteusecase

import com.itlab.domain.model.ContentItem
import com.itlab.domain.model.Note
import com.itlab.domain.repository.NotesRepository
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flowOf
import kotlinx.coroutines.flow.map

class SearchNotesUseCase(
    private val repo: NotesRepository,
    private val getUserIdUseCase: GetUserIdUseCase,
) {
    operator fun invoke(
        query: String,
        folderId: String? = null,
    ): Flow<List<Note>> {
        val userId = getUserIdUseCase()

        if (userId == null) {
            return flowOf(emptyList())
        }

        val normalizedQuery = query.trim().lowercase()
        return if (normalizedQuery.isBlank()) {
            repo.observeNotes(userId)
        } else {
            repo.observeNotes(userId).map { notes ->
                notes
                    .filter { note -> folderId == null || note.folderId == folderId }
                    .filter { note -> note.matches(normalizedQuery) }
            }
        }
    }

    private fun Note.matches(normalizedQuery: String): Boolean {
        val titleMatch = title.contains(normalizedQuery, ignoreCase = true)
        val contentMatch =
            contentItems.any { item ->
                item is ContentItem.Text && item.text.contains(normalizedQuery, ignoreCase = true)
            }

        return titleMatch || contentMatch
    }
}

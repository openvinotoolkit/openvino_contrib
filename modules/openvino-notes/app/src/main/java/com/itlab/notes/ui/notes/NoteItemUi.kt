package com.itlab.notes.ui.notes

import com.itlab.domain.model.ContentItem

data class NoteItemUi(
    val id: String,
    val userId: String,
    val title: String,
    val content: String,
    val folderId: String? = null,
    val attachments: List<ContentItem> = emptyList(),
    val isFavorite: Boolean = false,
    val tags: Set<String> = emptySet(),
    val summary: String? = null,
)

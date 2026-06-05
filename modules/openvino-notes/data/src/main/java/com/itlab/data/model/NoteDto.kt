package com.itlab.data.model

import kotlinx.serialization.Serializable

@Serializable
data class NoteDto(
    val id: String,
    val folderId: String?,
    val body: NoteBodyDto,
    val metadata: NoteMetaDto,
)

@Serializable
data class NoteBodyDto(
    val title: String,
    val content: String,
    val summary: String?,
)

@Serializable
data class NoteMetaDto(
    val createdAt: Long,
    val updatedAt: Long,
    val tags: String?,
    val isFavorite: Boolean,
)

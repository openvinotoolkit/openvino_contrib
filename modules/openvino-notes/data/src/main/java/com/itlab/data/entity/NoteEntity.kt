package com.itlab.data.entity

import androidx.room.Entity
import androidx.room.PrimaryKey
import kotlin.time.Instant

@Entity(tableName = "notes")
data class NoteEntity(
    @PrimaryKey
    val id: String,
    val userId: String,
    val title: String,
    val content: String,
    val folderId: String? = null,
    val createdAt: Instant,
    val updatedAt: Instant,
    val tags: String? = null,
    val isFavorite: Boolean = false,
    val isSynced: Boolean = false,
    val isDeleted: Boolean = false,
    val summary: String? = null,
)

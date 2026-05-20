package com.itlab.data.entity

import androidx.room.Entity
import androidx.room.ForeignKey
import androidx.room.Index
import androidx.room.PrimaryKey

@Entity(
    tableName = "media",
    indices = [Index(value = ["noteId"])],
    foreignKeys = [
        ForeignKey(
            entity = NoteEntity::class,
            parentColumns = ["id"],
            childColumns = ["noteId"],
            onDelete = ForeignKey.CASCADE,
        ),
    ],
)
data class MediaEntity(
    @PrimaryKey
    val id: String,
    val noteId: String,
    val type: String,
    val remoteUrl: String?,
    val localPath: String?,
    val mimeType: String,
    val isSynced: Boolean = false,
    val isDeleted: Boolean = false,
    val size: Long? = null,
)

package com.itlab.data.mapper

import com.itlab.data.entity.FolderEntity
import com.itlab.domain.model.NoteFolder

class NoteFolderMapper {
    fun toEntity(folder: NoteFolder): FolderEntity {
        val entityFolder =
            FolderEntity(
                id = folder.id,
                name = folder.name,
                createdAt = folder.createdAt,
                updatedAt = folder.updatedAt,
                metadata = folder.metadata,
                userId = folder.userId,
            )

        return entityFolder
    }

    fun toDomain(entity: FolderEntity): NoteFolder {
        val noteFolder =
            NoteFolder(
                id = entity.id,
                name = entity.name,
                createdAt = entity.createdAt,
                updatedAt = entity.updatedAt,
                metadata = entity.metadata,
                userId = entity.userId,
            )

        return noteFolder
    }
}

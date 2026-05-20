package com.itlab.data.mapper

import com.itlab.data.entity.FolderEntity
import com.itlab.data.model.FolderDto
import kotlinx.serialization.json.Json
import kotlin.time.Instant

class FolderEntityJsonConverter(
    private val json: Json =
        Json {
            ignoreUnknownKeys = true
            encodeDefaults = true
        },
) {
    fun FolderEntity.toDto(): FolderDto =
        FolderDto(
            id = id,
            name = name,
            createdAt = createdAt.toEpochMilliseconds(),
            updatedAt = updatedAt.toEpochMilliseconds(),
            metadata = metadata,
        )

    fun toEntity(
        jsonString: String,
        userId: String,
    ): FolderEntity {
        val dto = json.decodeFromString<FolderDto>(jsonString)
        return dto.toEntity(userId)
    }

    fun FolderEntity.toJson(): String {
        val dto = this.toDto()
        return json.encodeToString(dto)
    }

    fun FolderDto.toEntity(userId: String): FolderEntity =
        FolderEntity(
            id = id,
            userId = userId,
            name = name,
            createdAt = Instant.fromEpochMilliseconds(createdAt),
            updatedAt = Instant.fromEpochMilliseconds(updatedAt),
            isSynced = true,
            isDeleted = false,
            metadata = metadata,
        )
}

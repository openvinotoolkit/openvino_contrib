package com.itlab.data.mapper

import com.itlab.data.entity.MediaEntity
import com.itlab.data.entity.NoteEntity
import com.itlab.data.mapper.toDomain
import com.itlab.data.mapper.toDto
import com.itlab.data.model.ContentItemDto
import com.itlab.domain.model.ContentItem
import com.itlab.domain.model.Note
import com.itlab.domain.model.SyncState
import kotlinx.serialization.SerializationException
import kotlinx.serialization.json.Json
import timber.log.Timber

class NoteMapper(
    private val json: Json =
        Json {
            ignoreUnknownKeys = true
            encodeDefaults = true
        },
) {
    fun toEntities(note: Note): Pair<NoteEntity, List<MediaEntity>> {
        val noteId = note.id

        val mediaEntities =
            note.contentItems.mapNotNull { item ->
                toMediaEntity(item, noteId)
            }

        val noteEntity =
            NoteEntity(
                id = noteId,
                userId = note.userId,
                title = note.title,
                folderId = note.folderId,
                // Важно: сериализуем БЕЗ localPath, чтобы в JSON строки не зашивались абсолютные пути
                content = serializeContentWithoutLocalPaths(note.contentItems),
                createdAt = note.createdAt,
                updatedAt = note.updatedAt,
                tags = json.encodeToString(note.tags),
                isFavorite = note.isFavorite,
                isSynced = note.syncStatus == SyncState.SYNCED,
                summary = note.summary,
            )

        return noteEntity to mediaEntities
    }

    // Перегрузка старого метода для обратной совместимости, если где-то используется без медиа
    fun toDomain(entity: NoteEntity): Note = toDomain(entity, emptyList())

    // Основной рабочий метод для обогащения контента актуальными путями из БД
    fun toDomain(
        entity: NoteEntity,
        mediaEntities: List<MediaEntity>,
    ): Note {
        val rawItems =
            try {
                deserializeContent(entity.content)
            } catch (e: SerializationException) {
                Timber.e(e, "Note content mapping failed for entity: ${entity.id}")
                emptyList()
            }

        // Обогащаем элементы контента локальными путями и remoteUrl из таблицы MediaEntity
        val activeMedia = mediaEntities.filter { !it.isDeleted }
        val activeMediaIds = activeMedia.map { it.id }.toSet()
        val filterOrphanMedia = activeMedia.isNotEmpty()
        val enrichedItems =
            rawItems
                .filter { item ->
                    when (item) {
                        is ContentItem.Text -> true
                        is ContentItem.Image,
                        is ContentItem.File,
                        -> !filterOrphanMedia || item.id in activeMediaIds
                        else -> true
                    }
                }.map { item ->
                    val localMedia = activeMedia.find { it.id == item.id }
                    if (localMedia != null) {
                        when (item) {
                            is ContentItem.Image ->
                                item.copy(
                                    source =
                                        item.source.copy(
                                            localPath = localMedia.localPath,
                                            remoteUrl = localMedia.remoteUrl,
                                        ),
                                )
                            is ContentItem.File ->
                                item.copy(
                                    source =
                                        item.source.copy(
                                            localPath = localMedia.localPath,
                                            remoteUrl = localMedia.remoteUrl,
                                        ),
                                )
                            else -> item
                        }
                    } else {
                        item
                    }
                }

        val tags =
            try {
                json.decodeFromString<Set<String>>(entity.tags ?: "[]")
            } catch (e: SerializationException) {
                Timber.e(e, "Tags mapping failed for note ${entity.id}. Raw data: ${entity.tags}")
                emptySet()
            }

        return Note(
            id = entity.id,
            userId = entity.userId,
            title = entity.title,
            contentItems = enrichedItems,
            folderId = entity.folderId,
            createdAt = entity.createdAt,
            updatedAt = entity.updatedAt,
            tags = tags,
            isFavorite = entity.isFavorite,
            syncStatus = if (entity.isSynced) SyncState.SYNCED else SyncState.PENDING,
            summary = entity.summary,
        )
    }

    private fun toMediaEntity(
        item: ContentItem,
        noteId: String,
    ): MediaEntity? {
        val type: String
        val mimeType: String
        val source =
            when (item) {
                is ContentItem.Image -> {
                    type = "IMAGE"
                    mimeType = item.mimeType
                    item.source
                }
                is ContentItem.File -> {
                    type = "FILE"
                    mimeType = item.mimeType
                    item.source
                }
                else -> return null
            }

        return MediaEntity(
            id = item.id,
            noteId = noteId,
            type = type,
            remoteUrl = source.remoteUrl,
            localPath = source.localPath,
            mimeType = mimeType,
            size = (item as? ContentItem.File)?.size,
            isSynced = false,
            isDeleted = false,
        )
    }

    // Сбрасываем localPath в null перед упаковкой контента в JSON для БД/Облака
    private fun serializeContentWithoutLocalPaths(items: List<ContentItem>): String {
        val cleanedItems =
            items.map { item ->
                when (item) {
                    is ContentItem.Image -> item.copy(source = item.source.copy(localPath = null))
                    is ContentItem.File -> item.copy(source = item.source.copy(localPath = null))
                    else -> item
                }
            }
        val dtos = cleanedItems.map { it.toDto() }
        return json.encodeToString(dtos)
    }

    fun serializeContent(items: List<ContentItem>): String {
        val dtos = items.map { it.toDto() }
        return json.encodeToString(dtos)
    }

    /** Drops image/file blocks whose ids are not in [activeMediaIds] (e.g. after media delete). */
    fun pruneNoteContentJson(
        contentJson: String,
        activeMediaIds: Set<String>,
    ): String {
        val items =
            try {
                deserializeContent(contentJson)
            } catch (e: SerializationException) {
                Timber.e(e, "Cannot prune note content JSON")
                return contentJson
            }
        val pruned =
            items.filter { item ->
                when (item) {
                    is ContentItem.Text -> true
                    is ContentItem.Image,
                    is ContentItem.File,
                    -> item.id in activeMediaIds
                    else -> true
                }
            }
        if (pruned.size == items.size) return contentJson
        return serializeContentWithoutLocalPaths(pruned)
    }

    fun pruneNoteContentJsonRemovingIds(
        contentJson: String,
        mediaIdsToRemove: Set<String>,
    ): String {
        if (mediaIdsToRemove.isEmpty()) return contentJson
        val items =
            try {
                deserializeContent(contentJson)
            } catch (e: SerializationException) {
                Timber.e(e, "Cannot prune note content JSON")
                return contentJson
            }
        val pruned =
            items.filter { item ->
                when (item) {
                    is ContentItem.Text -> true
                    is ContentItem.Image,
                    is ContentItem.File,
                    -> item.id !in mediaIdsToRemove
                    else -> true
                }
            }
        if (pruned.size == items.size) return contentJson
        return serializeContentWithoutLocalPaths(pruned)
    }

    fun deserializeContent(jsonString: String): List<ContentItem> {
        val dtos = json.decodeFromString<List<ContentItemDto>>(jsonString)
        return dtos.map { it.toDomain() }
    }
}

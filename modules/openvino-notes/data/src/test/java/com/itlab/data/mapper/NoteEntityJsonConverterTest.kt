package com.itlab.data.mapper

import android.R.id
import com.itlab.data.entity.NoteEntity
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.time.Instant

class NoteEntityJsonConverterTest {
    private val converter = NoteEntityJsonConverter()
    private val userId = "user-123"

    @Test
    fun `toDoShouldCorrectlyMapAllFields`() {
        val entity =
            NoteEntity(
                id = "note-1",
                userId = userId,
                title = "Test Title",
                content = "Test Content",
                folderId = "folder-99",
                createdAt = Instant.fromEpochMilliseconds(1713950000000L),
                updatedAt = Instant.fromEpochMilliseconds(1713950005000L),
                tags = "work,important",
                isFavorite = true,
                isSynced = false,
                isDeleted = false,
                summary = "Brief summary",
            )
        val dto = with(converter) { entity.toDto() }

        assertEquals(entity.id, dto.id)
        assertEquals(entity.folderId, dto.folderId)
        assertEquals(entity.title, dto.body.title)
        assertEquals(entity.content, dto.body.content)
        assertEquals(entity.summary, dto.body.summary)
        assertEquals(1713950000000L, dto.metadata.createdAt)
        assertEquals(1713950005000L, dto.metadata.updatedAt)
        assertEquals(entity.tags, dto.metadata.tags)
        assertEquals(entity.isFavorite, dto.metadata.isFavorite)
    }

    @Test
    fun `toEntity should correctly parse JSON and set internal flags`() {
        val json =
            """
            {
                "id": "remote-id",
                "folderId": "folder-1",
                "body": {
                    "title": "Cloud Note",
                    "content": "Some content",
                    "summary": "Short"
                },
                "metadata": {
                    "createdAt": 1000,
                    "updatedAt": 2000,
                    "tags": "cloud",
                    "isFavorite": true
                }
            }
            """.trimIndent()

        val entity = converter.toEntity(json, userId)

        assertEquals("remote-id", entity.id)
        assertEquals(userId, entity.userId)
        assertEquals("Cloud Note", entity.title)
        assertEquals(Instant.fromEpochMilliseconds(1000), entity.createdAt)
        assertTrue("Заметка из JSON должна быть помечена как синхронизированная", entity.isSynced)
        assertEquals("Заметка из JSON не может быть помечена как удаленная", false, entity.isDeleted)
    }

    @Test
    fun `should handle null fields in mapping`() {
        val entity =
            NoteEntity(
                id = "note-nulls",
                userId = userId,
                title = "Title",
                content = "Content",
                folderId = null,
                createdAt = Instant.fromEpochMilliseconds(0),
                updatedAt = Instant.fromEpochMilliseconds(0),
                tags = null,
                isFavorite = false,
                isSynced = true,
                isDeleted = false,
                summary = null,
            )

        val dto = with(converter) { entity.toDto() }
        val json = with(converter) { entity.toJson() }
        val parsedEntity = converter.toEntity(json, userId)

        assertEquals(null, dto.folderId)
        assertEquals(null, dto.body.summary)
        assertEquals(null, dto.metadata.tags)

        assertEquals(null, parsedEntity.folderId)
        assertEquals(null, parsedEntity.summary)
        assertEquals(null, parsedEntity.tags)
    }
}

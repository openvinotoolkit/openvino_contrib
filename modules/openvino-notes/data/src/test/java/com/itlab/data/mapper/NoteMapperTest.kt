package com.itlab.data.mapper

import com.itlab.data.entity.NoteEntity
import com.itlab.domain.model.ContentItem
import com.itlab.domain.model.DataSource
import com.itlab.domain.model.Note
import kotlinx.serialization.json.Json
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.time.Instant

class NoteMapperTest {
    private val mapper = NoteMapper()
    private val testUserId = "test-user-id"
    private val testTime = Instant.parse("2026-03-24T12:00:00Z")

    @Test
    fun `toEntities should map all content types and media correctly`() {
        val note = createTestNoteWithMixedContent()

        val (entity, media) = mapper.toEntities(note)

        // Проверяем поля NoteEntity
        assertNoteEntityFields(note, entity)

        // Проверяем, что локальные пути корректно очищены в JSON контенте
        val decodedItems = mapper.deserializeContent(entity.content)
        val expectedCleanedItems =
            note.contentItems.map { item ->
                when (item) {
                    is ContentItem.Image -> item.copy(source = item.source.copy(localPath = null))
                    is ContentItem.File -> item.copy(source = item.source.copy(localPath = null))
                    else -> item
                }
            }
        assertEquals(expectedCleanedItems, decodedItems)

        // Проверяем маппинг сопутствующих медиафайлов
        assertMediaEntities(note.id, media)
    }

    private fun createTestNoteWithMixedContent() =
        Note(
            userId = testUserId,
            title = "Business",
            tags = setOf("money", "market"),
            contentItems =
                listOf(
                    ContentItem.Text(text = "I have money"),
                    ContentItem.Image(
                        source = DataSource(localPath = "local/path"),
                        mimeType = "image/png",
                    ),
                    ContentItem.File(
                        source = DataSource(remoteUrl = "https://cloud.com/doc"),
                        mimeType = "application/pdf",
                        name = "doc.pdf",
                        size = 1024L,
                    ),
                    ContentItem.Link(url = "https://google.com"),
                ),
            isFavorite = true,
            summary = "cars",
        )

    private fun assertNoteEntityFields(
        note: Note,
        entity: NoteEntity,
    ) {
        assertEquals(note.id, entity.id)
        assertEquals(testUserId, entity.userId)
        assertEquals("Business", entity.title)
        assertTrue(entity.isFavorite)
        assertFalse(entity.isSynced)
        assertEquals(null, entity.folderId)
        assertEquals(note.createdAt, entity.createdAt)
        assertEquals(note.updatedAt, entity.updatedAt)
        assertEquals(note.summary, entity.summary)
        assertEquals("[\"money\",\"market\"]", entity.tags)
    }

    private fun assertMediaEntities(
        noteId: String,
        media: List<com.itlab.data.entity.MediaEntity>,
    ) {
        assertEquals(2, media.size)

        val image = media.find { it.type == "IMAGE" }
        assertNotNull(image?.id)
        assertEquals(noteId, image?.noteId)
        assertEquals("local/path", image?.localPath)
        assertEquals(null, image?.remoteUrl)
        assertEquals("image/png", image?.mimeType)

        val file = media.find { it.type == "FILE" }
        assertNotNull(file?.id)
        assertEquals(noteId, file?.noteId)
        assertEquals("https://cloud.com/doc", file?.remoteUrl)
        assertEquals(null, file?.localPath)
        assertEquals("application/pdf", file?.mimeType)
        assertEquals(1024L, file?.size)
    }

    @Test
    fun `toDomain should return empty lists when JSON is corrupted`() {
        val corruptedEntity =
            NoteEntity(
                id = "test-id",
                userId = testUserId,
                title = "Broken Note",
                content = "!!not a json!!",
                tags = "{broken_tags}",
                createdAt = testTime,
                updatedAt = testTime,
            )

        val result = mapper.toDomain(corruptedEntity)

        assertTrue(result.contentItems.isEmpty())
        assertTrue(result.tags.isEmpty())
        assertEquals(testUserId, result.userId)
        assertEquals("Broken Note", result.title)
    }

    @Test
    fun `toDomain should correctly restore Note from NoteEntity`() {
        val originalItems =
            listOf(
                ContentItem.Text(text = "First item"),
                ContentItem.Link("https://itlab.com", "IT Lab"),
                ContentItem.Image(
                    source = DataSource(localPath = null, remoteUrl = "cloud/path"),
                    mimeType = "image/",
                ),
                ContentItem.File(
                    source = DataSource(localPath = null, remoteUrl = "https://cloud.com/doc"),
                    mimeType = "application/pdf",
                    name = "doc.pdf",
                    size = 1024L,
                ),
            )

        val originalTags = setOf("android", "testing")
        val json = Json { ignoreUnknownKeys = true }

        val entity =
            NoteEntity(
                id = "uuid-123",
                userId = testUserId,
                title = "Test Note",
                folderId = "fuid-100",
                content = mapper.serializeContent(originalItems),
                tags = json.encodeToString<Set<String>>(originalTags),
                isFavorite = true,
                createdAt = testTime,
                updatedAt = testTime,
            )

        val resultNote = mapper.toDomain(entity)

        assertEquals("uuid-123", resultNote.id)
        assertEquals(testUserId, resultNote.userId)
        assertEquals("Test Note", resultNote.title)
        assertEquals("fuid-100", resultNote.folderId)
        assertTrue(resultNote.isFavorite)
        assertEquals(originalItems, resultNote.contentItems)
        assertEquals(originalTags, resultNote.tags)
    }

    @Test
    fun `toDomain should handle null tags in NoteEntity by returning empty set`() {
        val entityWithNullTags =
            NoteEntity(
                id = "test-null-tags",
                userId = testUserId,
                title = "Note with NULL tags",
                content = "[]",
                tags = null,
                folderId = null,
                isFavorite = false,
                createdAt = testTime,
                updatedAt = testTime,
            )

        val resultNote = mapper.toDomain(entityWithNullTags)

        assertNotNull(resultNote.tags)
        assertTrue(resultNote.tags.isEmpty())
    }
}

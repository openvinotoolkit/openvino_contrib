package com.itlab.data.model

import kotlinx.serialization.json.Json
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Test

class ModelCoverageTest {
    private val json = Json { ignoreUnknownKeys = true }

    @Test
    fun testNoteDtoCoverage() {
        val body = NoteBodyDto("Title", "Content", "Summary")
        val meta = NoteMetaDto(1000L, 2000L, "tags", true)
        val note1 = NoteDto("1", "folder_1", body, meta)
        val note2 = note1.copy(id = "1")
        val note3 = note1.copy(id = "2", folderId = null)

        assertEquals(note1, note2)
        assertNotEquals(note1, note3)
        assertEquals(note1.hashCode(), note2.hashCode())
        assertNotNull(note1.toString())

        assertNull(note3.folderId)
        assertEquals("folder_1", note1.folderId)
    }

    @Test
    fun testContentItemDtoCoverage() {
        val testID = "testID"

        val text = ContentItemDto.Text(testID, "hello", TextFormatDto.MARKDOWN)
        val image = ContentItemDto.Image(testID, DataSourceDto(localPath = "/path"), "image/png", 100, 200)
        val file = ContentItemDto.File(testID, DataSourceDto(remoteUrl = "http"), "doc", "name", 1024L)
        val link = ContentItemDto.Link(testID, "url", null)

        assertEquals(text, text.copy())
        assertEquals(image, image.copy())
        assertEquals(file, file.copy())
        assertEquals(link, link.copy())

        assertNotEquals(text, image)
        assertNotEquals(text, file)
        assertNotEquals(text, link)
        val defaultText = ContentItemDto.Text(testID, "plain")
        assertEquals(TextFormatDto.PLAIN, defaultText.format)

        val emptySource = DataSourceDto()
        assertNull(emptySource.localPath)
        assertNull(emptySource.remoteUrl)
    }

    @Test
    fun testDataSourceDtoEquality() {
        val ds1 = DataSourceDto("local", "remote")
        val ds2 = DataSourceDto("local", "remote")
        val ds3 = DataSourceDto(null, null)

        assertEquals(ds1, ds2)
        assertNotEquals(ds1, ds3)
        assertFalse(ds1.equals("not a dto"))
        assertEquals(ds1.hashCode(), ds2.hashCode())
    }

    @Test
    fun testSerializationCoverage() {
        val testId = "testID"
        val items: List<ContentItemDto> =
            listOf(
                ContentItemDto.Text(testId, "test"),
                ContentItemDto.Text(testId, "md", TextFormatDto.MARKDOWN),
                ContentItemDto.Image(testId, DataSourceDto("path"), "png"),
                ContentItemDto.File(testId, DataSourceDto(remoteUrl = "url"), "pdf", "file.pdf", 100L),
                ContentItemDto.Link(testId, "http://test.com"),
            )

        items.forEach { item ->
            val string = json.encodeToString(item)
            val restored = json.decodeFromString<ContentItemDto>(string)

            assertEquals(item, restored)
            assertEquals(item.hashCode(), restored.hashCode())
            assertNotNull(item.toString())
        }
    }

    @Test
    fun testTextFormatEnum() {
        TextFormatDto.values().forEach { format ->
            val s = json.encodeToString(format)
            assertEquals(format, json.decodeFromString<TextFormatDto>(s))
        }
    }

    @Test
    fun testNoteDtoComplete() {
        val note =
            NoteDto(
                id = "id",
                folderId = null,
                body = NoteBodyDto("t", "c", null),
                metadata = NoteMetaDto(0, 0, null, false),
            )

        val s = json.encodeToString(note)
        val restored = json.decodeFromString<NoteDto>(s)

        assertEquals(note, restored)
        val copy = note.copy(folderId = "not_null")
        assertNotEquals(note, copy)
    }

    @Test
    fun testNoteMetaDtoFullCoverage() {
        val meta =
            NoteMetaDto(
                createdAt = 12345L,
                updatedAt = 67890L,
                tags = "work, urgent",
                isFavorite = true,
            )

        val jsonString = json.encodeToString(meta)
        val decoded = json.decodeFromString<NoteMetaDto>(jsonString)
        assertEquals(meta, decoded)

        val metaWithNull = meta.copy(tags = null)
        val nullJson = json.encodeToString(metaWithNull)
        val decodedNull = json.decodeFromString<NoteMetaDto>(nullJson)
        assertNull(decodedNull.tags)
        assertEquals(metaWithNull, decodedNull)

        assertNotNull(NoteMetaDto.serializer().descriptor)

        assertNotNull(meta.toString())
        assertEquals(meta.hashCode(), decoded.hashCode())
    }

    @Test
    fun testNoteMetaDtoHardcoreCoverage() {
        val meta = NoteMetaDto(1L, 2L, "tag", true)

        val serializer = NoteMetaDto.serializer()
        val jsonString = Json.encodeToString(serializer, meta)
        val restored = Json.decodeFromString(serializer, jsonString)

        val descriptor = serializer.descriptor
        for (i in 0 until descriptor.elementsCount) {
            assertNotNull(descriptor.getElementName(i))
            assertFalse(descriptor.isElementOptional(i))
        }

        val copy1 = meta.copy(createdAt = 10L)
        val copy2 = meta.copy(updatedAt = 20L)
        val copy3 = meta.copy(tags = "new")
        val copy4 = meta.copy(isFavorite = false)

        assertNotEquals(meta, copy1)
        assertNotEquals(meta, copy2)
        assertNotEquals(meta, copy3)
        assertNotEquals(meta, copy4)

        assertEquals(meta, restored)
        assertEquals(meta.hashCode(), restored.hashCode())
        assertNotNull(meta.toString())
    }

    @Test(expected = Exception::class)
    fun testNoteMetaSerializationError() {
        val brokenJson = """{"updatedAt":100, "isFavorite":true}"""
        Json.decodeFromString<NoteMetaDto>(brokenJson)
    }
}

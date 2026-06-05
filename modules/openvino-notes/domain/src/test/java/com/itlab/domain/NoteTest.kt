package com.itlab.domain

import com.itlab.domain.model.ContentItem
import com.itlab.domain.model.DataSource
import com.itlab.domain.model.Note
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class NoteTest {
    private val testUserId = "test_user_1"

    @Test
    fun note_creation() {
        val note = Note(title = "Hello", userId = testUserId)

        assertEquals("Hello", note.title)
        assertTrue(note.contentItems.isEmpty())
    }

    @Test
    fun note_copy() {
        val note = Note(title = "A", userId = testUserId)

        val updated = note.copy(title = "B")

        assertEquals("A", note.title)
        assertEquals("B", updated.title)
    }

    @Test
    fun content_items_types() {
        val items =
            listOf(
                ContentItem.Text(
                    text = "text",
                ),
                ContentItem.Image(
                    source = DataSource(localPath = "/cache/images/pic_001.jpg"),
                    mimeType = "image/png",
                ),
                ContentItem.Link(
                    url = "https://url.com",
                ),
                ContentItem.File(
                    source = DataSource(remoteUrl = "https://url.com"),
                    mimeType = "application/pdf",
                    name = "name",
                ),
            )

        assertTrue(items[0] is ContentItem.Text)
        assertTrue(items[1] is ContentItem.Image)
        assertTrue(items[2] is ContentItem.Link)
        assertTrue(items[3] is ContentItem.File)
    }
}

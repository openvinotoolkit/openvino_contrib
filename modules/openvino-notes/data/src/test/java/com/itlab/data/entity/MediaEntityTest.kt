package com.itlab.data.entity

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class MediaEntityTest {
    @Test
    fun `when MediaEntity is created, all properties map correctly`() {
        val media =
            MediaEntity(
                id = "media_1",
                noteId = "note_1",
                type = "image",
                remoteUrl = "s3://bucket/image.jpg",
                localPath = "test/path/image.jpg",
                mimeType = "image/jpeg",
                size = 2048L,
            )

        assertEquals("media_1", media.id)
        assertEquals("note_1", media.noteId)
        assertEquals("image", media.type)
        assertEquals("s3://bucket/image.jpg", media.remoteUrl)
        assertEquals("test/path/image.jpg", media.localPath)
        assertEquals("image/jpeg", media.mimeType)
        assertEquals(2048L, media.size)
    }

    @Test
    fun `when MediaEntity is created with null optional values, they are null`() {
        val media =
            MediaEntity(
                id = "media_2",
                noteId = "note_2",
                type = "audio",
                remoteUrl = "s3://bucket/audio.mp3",
                localPath = null,
                mimeType = "audio/mpeg",
            )

        assertNull(media.localPath)
        assertNull(media.size)
    }

    @Test
    fun `media equality`() {
        val m1 = MediaEntity("1", "n1", "img", "u", "p", "m")
        val m2 = MediaEntity("1", "n1", "img", "u", "p", "m")

        assertEquals(m1, m2)
        assertEquals(m1.hashCode(), m2.hashCode())
        assert(m1.toString().isNotEmpty())
    }

    @Test
    fun `media copy with null path`() {
        val media = MediaEntity("1", "n1", "img", "u", "path", "m")
        val noPathMedia = media.copy(localPath = null)

        assertEquals(null, noPathMedia.localPath)
    }
}

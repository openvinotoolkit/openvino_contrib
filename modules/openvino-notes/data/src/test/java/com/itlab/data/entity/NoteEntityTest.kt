package com.itlab.data.entity

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.time.Clock
import kotlin.time.Instant

class NoteEntityTest {
    val testTime = Instant.parse("2026-03-24T12:00:00Z")

    @Test
    fun `when NoteEntity is created with minimum args, default values are set correctly`() {
        val note =
            NoteEntity(
                id = "note_1",
                userId = "user_1",
                title = "Test Title",
                content = "Test Content",
                createdAt = testTime,
                updatedAt = testTime,
                summary = "about content",
            )

        assertEquals("note_1", note.id)
        assertEquals("user_1", note.userId)
        assertEquals("Test Title", note.title)
        assertEquals("Test Content", note.content)
        assertFalse(note.isSynced)
        assertEquals(testTime, note.createdAt)
        assertEquals(testTime, note.updatedAt)
        assertEquals(note.summary, "about content")
    }

    @Test fun `when NoteEntity is fully initialized, all fields match`() {
        val customTime = Clock.System.now()

        val note =
            NoteEntity(
                id = "note_2",
                userId = "user_2",
                title = "Title 2",
                content = "Content 2",
                createdAt = customTime,
                updatedAt = customTime,
                isSynced = true,
            )

        assertEquals("user_2", note.userId)
        assertEquals(customTime, note.createdAt)
        assertEquals(customTime, note.updatedAt)
        assertTrue(note.isSynced)
    }

    @Test
    fun `note creation and properties`() {
        val note =
            NoteEntity(
                id = "1",
                userId = "user_1",
                title = "Title",
                content = "Content",
                createdAt = testTime,
                updatedAt = testTime,
            )
        assertEquals("1", note.id)
        assertEquals("user_1", note.userId)
        assertEquals("Title", note.title)
        assertEquals(false, note.isSynced)
    }

    @Test
    fun `note equality and hashcode`() {
        val id = "1"
        val userId = "user_1"
        val title = "A"
        val content = "B"
        val timestamp = Instant.fromEpochMilliseconds(123456789L)

        val note1 =
            NoteEntity(
                id = id,
                userId = userId,
                title = title,
                content = content,
                isSynced = false,
                createdAt = timestamp,
                updatedAt = timestamp,
            )
        val note2 =
            NoteEntity(
                id = id,
                userId = userId,
                title = title,
                content = content,
                isSynced = false,
                createdAt = timestamp,
                updatedAt = timestamp,
            )

        assertEquals(note1, note2)
        assertEquals(note1.hashCode(), note2.hashCode())
    }

    @Test
    fun `note copy updates fields`() {
        val note =
            NoteEntity(
                "1",
                "user_1",
                "Old",
                "Text",
                createdAt = testTime,
                updatedAt = testTime,
            )
        val updated = note.copy(title = "New", isSynced = true)

        assertEquals("New", updated.title)
        assertEquals("user_1", updated.userId)
        assertEquals(true, updated.isSynced)
        assertEquals("1", updated.id)
    }
}

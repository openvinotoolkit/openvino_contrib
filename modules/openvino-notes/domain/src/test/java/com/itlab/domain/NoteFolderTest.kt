package com.itlab.domain

import com.itlab.domain.model.NoteFolder
import org.junit.Assert.assertEquals
import org.junit.Test

class NoteFolderTest {
    private val testUserId = "user_folder_test"

    @Test
    fun folder_creation() {
        val folder = NoteFolder(testUserId, id = "1", name = "Test")

        assertEquals("Test", folder.name)
    }

    @Test
    fun folder_copy() {
        val folder = NoteFolder(testUserId, id = "1", name = "Old")

        val updated = folder.copy(name = "New")

        assertEquals("Old", folder.name)
        assertEquals("New", updated.name)
    }
}

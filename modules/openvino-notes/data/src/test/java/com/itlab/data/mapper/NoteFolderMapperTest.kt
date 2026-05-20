package com.itlab.data.mapper

import com.itlab.data.entity.FolderEntity
import com.itlab.domain.model.NoteFolder
import org.junit.Assert.assertEquals
import org.junit.Test
import kotlin.time.Instant

class NoteFolderMapperTest {
    val mapper = NoteFolderMapper()
    val testTime = Instant.parse("2026-03-24T12:00:00Z")

    private val testUserId = "test_user_1"

    @Test
    fun `toEntity should map model correctly`() {
        val uiMetadata =
            mapOf(
                "color" to "#FF5733",
                "icon" to "folder_shared",
                "is_expanded" to "true",
                "display_mode" to "grid",
            )

        val noteFolder =
            NoteFolder(
                name = "Personal",
                createdAt = testTime,
                updatedAt = testTime,
                metadata = uiMetadata,
                userId = testUserId,
            )

        val entityFolder = mapper.toEntity(noteFolder)

        assertEquals(noteFolder.id, entityFolder.id)
        assertEquals(noteFolder.name, entityFolder.name)
        assertEquals(noteFolder.createdAt, entityFolder.createdAt)
        assertEquals(noteFolder.updatedAt, entityFolder.updatedAt)
        assertEquals(noteFolder.metadata, entityFolder.metadata)
        assertEquals(noteFolder.userId, entityFolder.userId)
    }

    @Test
    fun `toDomain should map entity correctly`() {
        val uiMetadata =
            mapOf(
                "color" to "#FF5733",
                "icon" to "folder_shared",
                "is_expanded" to "true",
                "display_mode" to "grid",
            )

        val entityFolder =
            FolderEntity(
                id = "test-id",
                name = "Personal",
                createdAt = testTime,
                updatedAt = testTime,
                metadata = uiMetadata,
                userId = testUserId,
            )

        val noteFolder = mapper.toDomain(entityFolder)

        assertEquals(noteFolder.id, entityFolder.id)
        assertEquals(noteFolder.name, entityFolder.name)
        assertEquals(noteFolder.createdAt, entityFolder.createdAt)
        assertEquals(noteFolder.updatedAt, entityFolder.updatedAt)
        assertEquals(noteFolder.metadata, entityFolder.metadata)
        assertEquals(noteFolder.userId, entityFolder.userId)
    }
}

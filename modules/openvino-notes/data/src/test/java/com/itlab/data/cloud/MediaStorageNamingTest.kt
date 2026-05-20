package com.itlab.data.cloud

import org.junit.Assert.assertEquals
import org.junit.Test

class MediaStorageNamingTest {
    @Test
    fun `compose and parse round trip for uuid ids`() {
        val noteId = "550e8400-e29b-41d4-a716-446655440000"
        val mediaId = "6ba7b810-9dad-11d1-80b4-00c04fd430c8"
        val name = MediaStorageNaming.compose(noteId, mediaId)
        assertEquals(noteId to mediaId, MediaStorageNaming.parse(name))
    }
}

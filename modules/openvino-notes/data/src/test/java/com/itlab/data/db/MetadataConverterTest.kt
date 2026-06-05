package com.itlab.data.db

import org.junit.Assert.assertEquals
import org.junit.Test

class MetadataConverterTest {
    private val converter = MetadataConverter()

    @Test
    fun `metadata map should be converted to json string and back`() {
        val originalMap = mapOf("color" to "blue", "priority" to "high")

        val jsonString = converter.fromMetadata(originalMap)
        val resultMap = converter.toMetadata(jsonString)

        assertEquals(originalMap, resultMap)
        assertEquals("blue", resultMap["color"])
    }

    @Test
    fun `invalid json should return empty map`() {
        val result = converter.toMetadata("invalid_json")
        assertEquals(0, result.size)
    }
}

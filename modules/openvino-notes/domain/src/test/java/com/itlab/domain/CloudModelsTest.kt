package com.itlab.domain

import com.itlab.domain.cloud.CloudMetadata
import com.itlab.domain.cloud.Result
import org.junit.Assert.assertEquals
import org.junit.Test
import kotlin.time.Instant

class CloudModelsTest {
    @Test
    fun `verify cloud note metadata model`() {
        val now = Instant.fromEpochMilliseconds(System.currentTimeMillis())
        val key = "notes/user1/note_123.json"

        val metadata = CloudMetadata(key = key, updatedAt = now)

        assertEquals(key, metadata.key)
        assertEquals(now, metadata.updatedAt)
    }

    @Test
    fun `verify cloud result success and error classes`() {
        val successData = "some_data"
        val success = Result.Success(successData)
        assertEquals(successData, success.data)

        val exception = RuntimeException("Cloud error")
        val error = Result.Error(exception)
        assertEquals(exception, error.exception)
    }
}

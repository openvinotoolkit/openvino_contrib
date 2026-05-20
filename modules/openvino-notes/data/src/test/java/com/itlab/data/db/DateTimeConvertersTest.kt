package com.itlab.data.db

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test
import kotlin.time.Instant

class DateTimeConvertersTest {
    private val converters = DateTimeConverters()

    @Test
    fun `fromTimestamp should convert long to instant and handle null`() {
        val timestamp = 1711272000000L
        val expected = Instant.fromEpochMilliseconds(timestamp)

        assertEquals(expected, converters.fromTimestamp(timestamp))
        assertNull(converters.fromTimestamp(null))
    }

    @Test
    fun `dateToTimestamp should convert instant to long and handle null`() {
        val timestamp = 1711272000000L
        val instant = Instant.fromEpochMilliseconds(timestamp)

        assertEquals(timestamp, converters.dateToTimestamp(instant))
        assertNull(converters.dateToTimestamp(null))
    }
}

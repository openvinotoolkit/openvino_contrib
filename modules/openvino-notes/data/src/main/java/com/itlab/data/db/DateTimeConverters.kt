package com.itlab.data.db

import androidx.room.TypeConverter
import kotlin.time.Instant

class DateTimeConverters {
    @TypeConverter
    fun fromTimestamp(value: Long?): Instant? =
        value?.let {
            Instant.fromEpochMilliseconds(it)
        }

    @TypeConverter
    fun dateToTimestamp(date: Instant?): Long? = date?.toEpochMilliseconds()
}

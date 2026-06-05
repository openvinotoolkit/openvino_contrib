package com.itlab.data.db

import androidx.room.TypeConverter
import kotlinx.serialization.SerializationException
import kotlinx.serialization.json.Json
import timber.log.Timber

class MetadataConverter {
    @TypeConverter
    fun fromMetadata(metadata: Map<String, String>): String = Json.encodeToString(metadata)

    @TypeConverter
    fun toMetadata(metadataString: String): Map<String, String> =
        try {
            Json.decodeFromString(metadataString)
        } catch (e: SerializationException) {
            Timber.e(e, "Failed to parse metadata")
            emptyMap()
        }
}

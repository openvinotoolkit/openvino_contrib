package com.itlab.domain.model

import java.util.UUID
import kotlin.time.Clock
import kotlin.time.Instant

data class Note(
    val userId: String,
    val id: String = UUID.randomUUID().toString(),
    val title: String = "",
    val folderId: String? = null,
    val contentItems: List<ContentItem> = emptyList(),
    val createdAt: Instant = Clock.System.now(),
    val updatedAt: Instant = Clock.System.now(),
    val tags: Set<String> = emptySet(),
    val isFavorite: Boolean = false,
    val summary: String? = null,
    val syncStatus: SyncState = SyncState.PENDING,
)

data class DataSource(
    val localPath: String? = null,
    val remoteUrl: String? = null,
) {
    val displayPath: String? get() = localPath ?: remoteUrl
}

sealed class ContentItem {
    abstract val id: String

    data class Text(
        override val id: String = UUID.randomUUID().toString(),
        val text: String,
        val format: TextFormat = TextFormat.PLAIN,
    ) : ContentItem()

    data class Image(
        override val id: String = UUID.randomUUID().toString(),
        val source: DataSource,
        val mimeType: String,
        val width: Int? = null,
        val height: Int? = null,
    ) : ContentItem()

    data class File(
        override val id: String = UUID.randomUUID().toString(),
        val source: DataSource,
        val mimeType: String,
        val name: String,
        val size: Long? = null,
    ) : ContentItem()

    data class Link(
        override val id: String = UUID.randomUUID().toString(),
        val url: String,
        val title: String? = null,
    ) : ContentItem()
}

enum class TextFormat {
    PLAIN,
    MARKDOWN,
    HTML,
}

enum class SyncState {
    SYNCED,
    PENDING,
    SYNCING,
    ERROR,
}

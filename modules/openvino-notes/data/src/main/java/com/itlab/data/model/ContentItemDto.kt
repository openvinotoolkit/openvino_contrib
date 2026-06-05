package com.itlab.data.model

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

@Serializable
sealed class ContentItemDto {
    abstract val id: String

    @Serializable
    data class Text(
        override val id: String,
        val text: String,
        val format: TextFormatDto = TextFormatDto.PLAIN,
    ) : ContentItemDto()

    @Serializable
    data class Image(
        override val id: String,
        val source: DataSourceDto,
        val mimeType: String,
        val width: Int? = null,
        val height: Int? = null,
    ) : ContentItemDto()

    @Serializable
    data class File(
        override val id: String,
        val source: DataSourceDto,
        val mimeType: String,
        val name: String,
        val size: Long? = null,
    ) : ContentItemDto()

    @Serializable
    data class Link(
        override val id: String,
        val url: String,
        val title: String? = null,
    ) : ContentItemDto()
}

@Serializable
enum class TextFormatDto {
    @SerialName("PLAIN")
    PLAIN,

    @SerialName("MARKDOWN")
    MARKDOWN,

    @SerialName("HTML")
    HTML,
}

@Serializable
data class DataSourceDto(
    val localPath: String? = null,
    val remoteUrl: String? = null,
)

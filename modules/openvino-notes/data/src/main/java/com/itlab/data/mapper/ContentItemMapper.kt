package com.itlab.data.mapper

import com.itlab.data.model.ContentItemDto
import com.itlab.data.model.DataSourceDto
import com.itlab.data.model.TextFormatDto
import com.itlab.domain.model.ContentItem
import com.itlab.domain.model.DataSource
import com.itlab.domain.model.TextFormat

fun ContentItem.toDto(): ContentItemDto =
    when (this) {
        is ContentItem.Text -> ContentItemDto.Text(id, text, format.toDto())
        is ContentItem.Image -> ContentItemDto.Image(id, source.toDto(), mimeType, width, height)
        is ContentItem.File -> ContentItemDto.File(id, source.toDto(), mimeType, name, size)
        is ContentItem.Link -> ContentItemDto.Link(id, url, title)
    }

fun ContentItemDto.toDomain(): ContentItem =
    when (this) {
        is ContentItemDto.Text -> ContentItem.Text(id, text, format.toDomain())
        is ContentItemDto.Image -> ContentItem.Image(id, source.toDomain(), mimeType, width, height)
        is ContentItemDto.File -> ContentItem.File(id, source.toDomain(), mimeType, name, size)
        is ContentItemDto.Link -> ContentItem.Link(id, url, title)
    }

fun DataSource.toDto() = DataSourceDto(localPath, remoteUrl)

fun DataSourceDto.toDomain() = DataSource(localPath, remoteUrl)

fun TextFormat.toDto() = TextFormatDto.valueOf(this.name)

fun TextFormatDto.toDomain() = TextFormat.valueOf(this.name)

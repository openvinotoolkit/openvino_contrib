package com.itlab.domain.usecase.contentusecase

import com.itlab.domain.model.ContentItem
import java.util.UUID

class CreateContentItemUseCase {
    operator fun invoke(item: ContentItem): ContentItem =
        when (item) {
            is ContentItem.Text -> item.copy(id = UUID.randomUUID().toString())
            is ContentItem.Image -> item.copy(id = UUID.randomUUID().toString())
            is ContentItem.File -> item.copy(id = UUID.randomUUID().toString())
            is ContentItem.Link -> item.copy(id = UUID.randomUUID().toString())
        }
}

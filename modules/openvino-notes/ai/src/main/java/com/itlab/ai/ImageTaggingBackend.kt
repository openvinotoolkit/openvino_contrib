package com.itlab.ai

interface ImageTaggingBackend {
    fun release() = Unit

    suspend fun tagImages(imageSources: List<String>): Set<String>
}

class UnavailableImageTaggingBackend : ImageTaggingBackend {
    override suspend fun tagImages(imageSources: List<String>): Set<String> = emptySet()
}

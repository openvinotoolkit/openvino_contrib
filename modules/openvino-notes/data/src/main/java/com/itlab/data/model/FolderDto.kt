package com.itlab.data.model

import kotlinx.serialization.Serializable

@Serializable
data class FolderDto(
    val id: String,
    val name: String,
    val createdAt: Long,
    val updatedAt: Long,
    val metadata: Map<String, String> = emptyMap(),
)

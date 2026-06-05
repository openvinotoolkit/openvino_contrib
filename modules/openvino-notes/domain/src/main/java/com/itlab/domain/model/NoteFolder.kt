package com.itlab.domain.model

import java.util.UUID
import kotlin.time.Clock
import kotlin.time.Instant

data class NoteFolder(
    val userId: String,
    val id: String = UUID.randomUUID().toString(),
    val name: String,
    val createdAt: Instant = Clock.System.now(),
    val updatedAt: Instant = Clock.System.now(),
    val metadata: Map<String, String> = emptyMap(),
)

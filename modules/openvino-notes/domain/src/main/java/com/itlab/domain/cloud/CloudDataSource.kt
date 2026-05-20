package com.itlab.domain.cloud

import kotlin.time.Instant

@JvmInline
value class DomainFile(
    val path: String,
)

sealed interface Result<out T> {
    data class Success<out T>(
        val data: T,
    ) : Result<T>

    data class Error(
        val exception: Throwable,
    ) : Result<Nothing>
}

interface CloudDataSource {
    suspend fun listNoteMetadata(userId: String): Result<List<CloudMetadata>>

    suspend fun listMediaMetadata(userId: String): Result<List<CloudMediaMetadata>>

    suspend fun listFolderMetadata(userId: String): Result<List<CloudMetadata>>

    suspend fun downloadFolder(key: String): Result<String>

    suspend fun uploadFolder(
        key: String,
        json: String,
    ): Result<Unit>

    suspend fun downloadNote(key: String): Result<String>

    suspend fun uploadNote(
        key: String,
        json: String,
    ): Result<Unit>

    suspend fun uploadMedia(
        key: String,
        file: DomainFile,
        mimeType: String,
    ): Result<Unit>

    suspend fun downloadMedia(
        key: String,
        destination: DomainFile,
    ): Result<Unit>

    suspend fun deleteObject(key: String): Result<Unit>
}

data class CloudMetadata(
    val key: String,
    val updatedAt: Instant,
)

data class CloudMediaMetadata(
    val key: String,
    val mediaId: String,
    val mimeType: String,
)

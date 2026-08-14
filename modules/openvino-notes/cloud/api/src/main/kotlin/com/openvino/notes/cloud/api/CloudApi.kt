// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package com.openvino.notes.cloud.api

import com.openvino.notes.kernel.AccountKey
import java.time.Instant

@JvmInline value class RemoteObjectId(val value: String)
@JvmInline value class RemoteCursor(val value: String)
@JvmInline value class RemoteRevision(val value: String)
@JvmInline value class TransferSessionId(val value: String)

data class RemoteObject(
    val id: RemoteObjectId,
    val name: String,
    val mediaType: String,
    val modifiedAt: Instant,
    val bytes: ByteArray,
) {
    override fun equals(other: Any?): Boolean = other is RemoteObject && id == other.id && bytes.contentEquals(other.bytes)
    override fun hashCode(): Int = 31 * id.hashCode() + bytes.contentHashCode()
}

data class RemoteObjectMetadata(
    val id: RemoteObjectId,
    val name: String,
    val mediaType: String,
    val modifiedAt: Instant,
    val revision: RemoteRevision,
    val deleted: Boolean = false,
)

data class RemoteChangePage(
    val changes: List<RemoteObjectMetadata>,
    val nextCursor: RemoteCursor?,
    val resetRequired: Boolean = false,
)

enum class RemoteErrorCode {
    NOT_CONFIGURED,
    SIGNED_OUT,
    NOT_AUTHORIZED,
    NOT_FOUND,
    CONFLICT,
    UNAVAILABLE,
    INVALID_CURSOR,
    INVALID_RESPONSE,
}
data class RemoteError(val code: RemoteErrorCode, val diagnosticCode: String)

sealed interface RemoteOutcome<out T> {
    data class Success<T>(val value: T) : RemoteOutcome<T>
    data class Failure(val error: RemoteError) : RemoteOutcome<Nothing>
}

interface RemoteObjectStore {
    suspend fun list(accountKey: AccountKey): RemoteOutcome<List<RemoteObjectMetadata>>

    suspend fun put(
        accountKey: AccountKey,
        objectValue: RemoteObject,
        expectedRevision: RemoteRevision? = null,
    ): RemoteOutcome<RemoteObjectMetadata>

    suspend fun get(accountKey: AccountKey, id: RemoteObjectId): RemoteOutcome<RemoteObject>

    suspend fun delete(
        accountKey: AccountKey,
        id: RemoteObjectId,
        expectedRevision: RemoteRevision? = null,
    ): RemoteOutcome<RemoteObjectMetadata>
}

interface RemoteChangeFeed {
    suspend fun startCursor(accountKey: AccountKey): RemoteOutcome<RemoteCursor>
    suspend fun changes(accountKey: AccountKey, cursor: RemoteCursor): RemoteOutcome<RemoteChangePage>
}

data class UploadDescriptor(
    val objectId: RemoteObjectId,
    val name: String,
    val mediaType: String,
    val sizeBytes: Long,
)

data class UploadSession(
    val id: TransferSessionId,
    val objectId: RemoteObjectId,
    val nextOffset: Long,
    val expiresAt: Instant? = null,
)

class UploadChunk(val offset: Long, val bytes: ByteArray, val final: Boolean) {
    override fun equals(other: Any?): Boolean = other is UploadChunk &&
        offset == other.offset && final == other.final && bytes.contentEquals(other.bytes)

    override fun hashCode(): Int = 31 * (31 * offset.hashCode() + bytes.contentHashCode()) + final.hashCode()
}

fun interface RemoteByteSink {
    suspend fun write(bytes: ByteArray)
}

/** Streaming boundary for large media; callers choose chunk size and never need one full in-memory object. */
interface ResumableTransferClient {
    suspend fun startUpload(
        accountKey: AccountKey,
        descriptor: UploadDescriptor,
        expectedRevision: RemoteRevision? = null,
    ): RemoteOutcome<UploadSession>

    suspend fun resumeUpload(accountKey: AccountKey, sessionId: TransferSessionId): RemoteOutcome<UploadSession>
    suspend fun uploadChunk(
        accountKey: AccountKey,
        sessionId: TransferSessionId,
        chunk: UploadChunk,
    ): RemoteOutcome<UploadSession>

    suspend fun downloadStream(
        accountKey: AccountKey,
        id: RemoteObjectId,
        sink: RemoteByteSink,
    ): RemoteOutcome<RemoteObjectMetadata>
}

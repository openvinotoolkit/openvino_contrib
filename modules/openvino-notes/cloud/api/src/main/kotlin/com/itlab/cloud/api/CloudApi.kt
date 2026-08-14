package com.itlab.cloud.api

import com.itlab.identity.api.AccountId
import java.time.Instant

@JvmInline value class RemoteObjectId(val value: String)
@JvmInline value class RemoteCursor(val value: String)

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
    val deleted: Boolean = false,
)

data class RemoteChangePage(val changes: List<RemoteObjectMetadata>, val nextCursor: RemoteCursor?)

enum class RemoteErrorCode { NOT_CONFIGURED, SIGNED_OUT, NOT_AUTHORIZED, NOT_FOUND, CONFLICT, UNAVAILABLE, INVALID_RESPONSE }
data class RemoteError(val code: RemoteErrorCode, val diagnosticCode: String)

sealed interface RemoteOutcome<out T> {
    data class Success<T>(val value: T) : RemoteOutcome<T>
    data class Failure(val error: RemoteError) : RemoteOutcome<Nothing>
}

interface RemoteObjectStore {
    suspend fun put(accountId: AccountId, objectValue: RemoteObject): RemoteOutcome<RemoteObjectMetadata>
    suspend fun get(accountId: AccountId, id: RemoteObjectId): RemoteOutcome<RemoteObject>
    suspend fun delete(accountId: AccountId, id: RemoteObjectId): RemoteOutcome<Unit>
}

interface RemoteChangeFeed {
    suspend fun changes(accountId: AccountId, cursor: RemoteCursor?): RemoteOutcome<RemoteChangePage>
}

// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package com.openvino.notes.notes.api.port

import com.openvino.notes.kernel.AccountKey
import com.openvino.notes.notes.api.AttachmentId
import com.openvino.notes.notes.api.AttachmentMetadata
import com.openvino.notes.notes.api.Folder
import com.openvino.notes.notes.api.FolderId
import com.openvino.notes.notes.api.Note
import com.openvino.notes.notes.api.NoteId
import java.io.ByteArrayOutputStream
import java.time.Instant
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.flow.Flow

interface NotesRepository {
    fun observe(accountKey: AccountKey): Flow<List<Note>>
    suspend fun find(accountKey: AccountKey, id: NoteId): Note?
    suspend fun countInFolder(accountKey: AccountKey, folderId: FolderId): Int
    suspend fun save(note: Note)
    suspend fun delete(accountKey: AccountKey, id: NoteId): Boolean
}

interface FolderRepository {
    fun observe(accountKey: AccountKey): Flow<List<Folder>>
    suspend fun find(accountKey: AccountKey, id: FolderId): Folder?
    suspend fun findByName(accountKey: AccountKey, name: String): Folder?
    suspend fun save(folder: Folder)
    suspend fun delete(accountKey: AccountKey, id: FolderId): Boolean
}

@JvmInline value class RemoteRevision(val value: String)

enum class LocalChangeKind { UPSERT, DELETE }

data class LocalNoteChange(
    val changeId: String,
    val accountKey: AccountKey,
    val noteId: NoteId,
    val kind: LocalChangeKind,
    val baseRevision: RemoteRevision?,
    val changedAt: Instant,
    val payload: Note? = null,
)

sealed interface RemoteNoteChange {
    val noteId: NoteId?

    data class Upsert(
        val note: Note,
        val baseRevision: RemoteRevision?,
        val revision: RemoteRevision,
    ) : RemoteNoteChange {
        override val noteId: NoteId = note.id
    }

    data class Tombstone(
        override val noteId: NoteId,
        val revision: RemoteRevision,
    ) : RemoteNoteChange

    data class Malformed(
        override val noteId: NoteId?,
        val diagnosticCode: String,
    ) : RemoteNoteChange
}

sealed interface RemoteApplyResult {
    val noteId: NoteId?
    data class Applied(override val noteId: NoteId, val revision: RemoteRevision) : RemoteApplyResult
    data class Merged(override val noteId: NoteId, val generatedChangeId: String) : RemoteApplyResult
    data class Conflict(
        override val noteId: NoteId,
        val localRevision: RemoteRevision?,
        val remoteRevision: RemoteRevision,
    ) : RemoteApplyResult
    data class TombstoneApplied(override val noteId: NoteId, val revision: RemoteRevision) : RemoteApplyResult
    data class RejectedMalformed(override val noteId: NoteId?, val diagnosticCode: String) : RemoteApplyResult
}

data class LocalFolderChange(
    val changeId: String,
    val accountKey: AccountKey,
    val folderId: FolderId,
    val kind: LocalChangeKind,
    val baseRevision: RemoteRevision?,
    val changedAt: Instant,
    val payload: Folder? = null,
)

sealed interface RemoteFolderChange {
    val folderId: FolderId?

    data class Upsert(
        val folder: Folder,
        val baseRevision: RemoteRevision?,
        val revision: RemoteRevision,
    ) : RemoteFolderChange {
        override val folderId: FolderId = folder.id
    }

    data class Tombstone(
        override val folderId: FolderId,
        val revision: RemoteRevision,
    ) : RemoteFolderChange

    data class Malformed(
        override val folderId: FolderId?,
        val diagnosticCode: String,
    ) : RemoteFolderChange
}

sealed interface FolderRemoteApplyResult {
    val folderId: FolderId?
    data class Applied(override val folderId: FolderId, val revision: RemoteRevision) : FolderRemoteApplyResult
    data class Conflict(
        override val folderId: FolderId,
        val localRevision: RemoteRevision?,
        val remoteRevision: RemoteRevision,
    ) : FolderRemoteApplyResult
    data class TombstoneApplied(override val folderId: FolderId, val revision: RemoteRevision) : FolderRemoteApplyResult
    data class RejectedMalformed(override val folderId: FolderId?, val diagnosticCode: String) : FolderRemoteApplyResult
}

interface NotesSyncPort {
    suspend fun pendingChanges(accountKey: AccountKey, limit: Int = 100): List<LocalNoteChange>
    suspend fun acknowledge(accountKey: AccountKey, changeIds: Set<String>)
    suspend fun applyRemote(accountKey: AccountKey, changes: List<RemoteNoteChange>): List<RemoteApplyResult>
}

interface FolderSyncPort {
    suspend fun pendingFolderChanges(accountKey: AccountKey, limit: Int = 100): List<LocalFolderChange>
    suspend fun acknowledgeFolderChanges(accountKey: AccountKey, changeIds: Set<String>)
    suspend fun applyRemoteFolders(
        accountKey: AccountKey,
        changes: List<RemoteFolderChange>,
    ): List<FolderRemoteApplyResult>
}

/** A stable-size random-access binary snapshot suitable for bounded reads and resumable transfer. */
interface BinarySource {
    val sizeBytes: Long

    /** Returns up to [maxBytes] from [offset], or an empty array at end of source. */
    suspend fun read(offset: Long, maxBytes: Int): ByteArray
}

class AttachmentContentConflictException(val attachmentId: AttachmentId) :
    IllegalStateException("AttachmentId already identifies different binary content")

interface AttachmentContentPort {
    /**
     * Stores immutable content. Repeating an identical write is idempotent; different bytes for the same
     * [AttachmentMetadata.id] must fail with [AttachmentContentConflictException].
     */
    suspend fun put(accountKey: AccountKey, attachment: AttachmentMetadata, source: BinarySource)
    suspend fun open(accountKey: AccountKey, attachmentId: AttachmentId): BinarySource?
    suspend fun delete(accountKey: AccountKey, attachmentId: AttachmentId): Boolean
}

fun binarySourceOf(bytes: ByteArray): BinarySource {
    val snapshot = bytes.copyOf()
    return object : BinarySource {
        override val sizeBytes: Long = snapshot.size.toLong()

        override suspend fun read(offset: Long, maxBytes: Int): ByteArray {
            require(offset >= 0) { "offset must not be negative" }
            require(maxBytes > 0) { "maxBytes must be positive" }
            if (offset >= sizeBytes) return byteArrayOf()
            val start = offset.toInt()
            val end = minOf(snapshot.size.toLong(), offset + maxBytes.toLong()).toInt()
            return snapshot.copyOfRange(start, end)
        }
    }
}

/** Explicit bounded helper for consumers, such as image inference, that require a contiguous payload. */
suspend fun BinarySource.readAll(maxTotalBytes: Long): ByteArray {
    require(maxTotalBytes >= 0) { "maxTotalBytes must not be negative" }
    require(sizeBytes >= 0) { "Binary source size must not be negative" }
    require(sizeBytes <= maxTotalBytes) { "Binary source exceeds the allowed size" }
    require(sizeBytes <= Int.MAX_VALUE) { "Binary source is too large for a contiguous byte array" }
    val output = ByteArrayOutputStream(minOf(sizeBytes, COPY_CHUNK_BYTES.toLong()).toInt())
    var offset = 0L
    while (offset < sizeBytes) {
        currentCoroutineContext().ensureActive()
        val requested = minOf(COPY_CHUNK_BYTES.toLong(), sizeBytes - offset).toInt()
        val chunk = read(offset, requested)
        check(chunk.isNotEmpty()) { "Binary source ended before its declared size" }
        check(chunk.size <= requested) { "Binary source returned more bytes than requested" }
        output.write(chunk)
        offset += chunk.size
    }
    return output.toByteArray()
}

private const val COPY_CHUNK_BYTES = 64 * 1024

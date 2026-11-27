// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package com.openvino.notes.cloud.api

import com.openvino.notes.kernel.AccountKey
import java.io.ByteArrayOutputStream
import java.time.Instant

class FakeRemoteStore :
    RemoteObjectStore,
    RemoteChangeFeed,
    ResumableTransferClient {
    val objects = linkedMapOf<Pair<AccountKey, RemoteObjectId>, RemoteObject>()
    private val revisions = mutableMapOf<Pair<AccountKey, RemoteObjectId>, Long>()
    private val uploads = mutableMapOf<TransferSessionId, FakeUpload>()
    private var nextUploadId = 0L
    var changeOutcome: RemoteOutcome<RemoteChangePage> = RemoteOutcome.Success(RemoteChangePage(emptyList(), null))
    var startCursorOutcome: RemoteOutcome<RemoteCursor> = RemoteOutcome.Success(RemoteCursor("0"))

    override suspend fun list(accountKey: AccountKey): RemoteOutcome<List<RemoteObjectMetadata>> =
        RemoteOutcome.Success(
            objects.filterKeys { it.first == accountKey }.map { (key, value) ->
                value.metadata(RemoteRevision(revisions.getValue(key).toString()))
            },
        )

    override suspend fun put(
        accountKey: AccountKey,
        objectValue: RemoteObject,
        expectedRevision: RemoteRevision?,
    ): RemoteOutcome<RemoteObjectMetadata> {
        val key = accountKey to objectValue.id
        val current = revisions[key]
        if (expectedRevision != null && expectedRevision.value != current?.toString()) {
            return conflict()
        }
        val revision = (current ?: 0L) + 1L
        objects[key] = objectValue
        revisions[key] = revision
        return RemoteOutcome.Success(objectValue.metadata(RemoteRevision(revision.toString())))
    }

    override suspend fun get(
        accountKey: AccountKey,
        id: RemoteObjectId,
    ): RemoteOutcome<RemoteObject> =
        objects[accountKey to id]?.let { RemoteOutcome.Success(it) }
            ?: RemoteOutcome.Failure(RemoteError(RemoteErrorCode.NOT_FOUND, "fake.not_found"))

    override suspend fun delete(
        accountKey: AccountKey,
        id: RemoteObjectId,
        expectedRevision: RemoteRevision?,
    ): RemoteOutcome<RemoteObjectMetadata> {
        val key = accountKey to id
        val current = revisions[key]
        if (expectedRevision != null && expectedRevision.value != current?.toString()) {
            return conflict()
        }
        val objectValue =
            objects.remove(key)
                ?: return RemoteOutcome.Failure(RemoteError(RemoteErrorCode.NOT_FOUND, "fake.not_found"))
        val revision = (current ?: 0L) + 1L
        revisions[key] = revision
        return RemoteOutcome.Success(objectValue.metadata(RemoteRevision(revision.toString()), deleted = true))
    }

    override suspend fun startCursor(accountKey: AccountKey): RemoteOutcome<RemoteCursor> = startCursorOutcome

    override suspend fun changes(
        accountKey: AccountKey,
        cursor: RemoteCursor,
    ): RemoteOutcome<RemoteChangePage> = changeOutcome

    override suspend fun startUpload(
        accountKey: AccountKey,
        descriptor: UploadDescriptor,
        expectedRevision: RemoteRevision?,
    ): RemoteOutcome<UploadSession> {
        val session =
            UploadSession(
                TransferSessionId("fake-upload-${++nextUploadId}"),
                descriptor.objectId,
                nextOffset = 0,
                expiresAt = Instant.now().plusSeconds(3600),
            )
        uploads[session.id] = FakeUpload(accountKey, descriptor, expectedRevision, session)
        return RemoteOutcome.Success(session)
    }

    override suspend fun resumeUpload(
        accountKey: AccountKey,
        sessionId: TransferSessionId,
    ): RemoteOutcome<ResumableUploadState> {
        val upload =
            uploads[sessionId]?.takeIf { it.accountKey == accountKey }
                ?: return notFound("fake.upload_session_not_found")

        if (upload.isSessionExpired) {
            return RemoteOutcome.Success(ResumableUploadState.Expired(upload.session))
        }

        val metadata = upload.completedMetadata
        if (metadata != null) {
            return RemoteOutcome.Success(ResumableUploadState.Completed(metadata))
        }

        return RemoteOutcome.Success(ResumableUploadState.Active(upload.session))
    }

    override suspend fun uploadChunk(
        accountKey: AccountKey,
        sessionId: TransferSessionId,
        chunk: UploadChunk,
    ): RemoteOutcome<UploadChunkOutcome> {
        val upload =
            uploads[sessionId]?.takeIf { it.accountKey == accountKey }
                ?: return notFound("fake.upload_session_not_found")

        if (upload.isSessionExpired) {
            return RemoteOutcome.Failure(RemoteError(RemoteErrorCode.NOT_AUTHORIZED, "Session expired"))
        }

        val completed = upload.completedMetadata
        if (completed != null) {
            return RemoteOutcome.Success(UploadChunkOutcome.Completed(completed))
        }

        if (chunk.offset != upload.session.nextOffset) return invalidTransfer("fake.upload_offset_mismatch")
        if (chunk.bytes.size.toLong() > upload.descriptor.sizeBytes - chunk.offset) {
            return invalidTransfer("fake.upload_exceeds_declared_size")
        }
        upload.content.write(chunk.bytes)
        upload.session = upload.session.copy(nextOffset = chunk.offset + chunk.bytes.size)
        if (!chunk.final) return RemoteOutcome.Success(UploadChunkOutcome.InProgress(upload.session))
        if (upload.session.nextOffset != upload.descriptor.sizeBytes) {
            return invalidTransfer("fake.upload_incomplete")
        }
        val objectValue =
            RemoteObject(
                id = upload.descriptor.objectId,
                name = upload.descriptor.name,
                mediaType = upload.descriptor.mediaType,
                modifiedAt = Instant.now(),
                bytes = upload.content.toByteArray(),
            )
        return when (val outcome = put(accountKey, objectValue, upload.expectedRevision)) {
            is RemoteOutcome.Failure -> {
                outcome
            }

            is RemoteOutcome.Success -> {
                upload.completedMetadata = outcome.value
                RemoteOutcome.Success(UploadChunkOutcome.Completed(outcome.value))
            }
        }
    }

    override suspend fun downloadStream(
        accountKey: AccountKey,
        id: RemoteObjectId,
        sink: RemoteByteSink,
    ): RemoteOutcome<RemoteObjectMetadata> {
        val value =
            objects[accountKey to id]
                ?: return RemoteOutcome.Failure(RemoteError(RemoteErrorCode.NOT_FOUND, "fake.not_found"))
        var offset = 0
        while (offset < value.bytes.size) {
            val end = minOf(value.bytes.size, offset + TRANSFER_CHUNK_BYTES)
            sink.write(value.bytes.copyOfRange(offset, end))
            offset = end
        }
        return RemoteOutcome.Success(value.metadata(RemoteRevision(revisions.getValue(accountKey to id).toString())))
    }

    private fun conflict(): RemoteOutcome.Failure = RemoteOutcome.Failure(RemoteError(RemoteErrorCode.CONFLICT, "fake.revision_conflict"))

    private fun notFound(code: String): RemoteOutcome.Failure = RemoteOutcome.Failure(RemoteError(RemoteErrorCode.NOT_FOUND, code))

    private fun invalidTransfer(code: String): RemoteOutcome.Failure =
        RemoteOutcome.Failure(RemoteError(RemoteErrorCode.INVALID_RESPONSE, code))

    private fun RemoteObject.metadata(
        revision: RemoteRevision,
        deleted: Boolean = false,
    ) = RemoteObjectMetadata(
        id = id,
        name = name,
        mediaType = mediaType,
        modifiedAt = if (deleted) Instant.now() else modifiedAt,
        revision = revision,
        deleted = deleted,
    )

    private data class FakeUpload(
        val accountKey: AccountKey,
        val descriptor: UploadDescriptor,
        val expectedRevision: RemoteRevision?,
        var session: UploadSession,
        var completedMetadata: RemoteObjectMetadata? = null,
        val content: ByteArrayOutputStream = ByteArrayOutputStream(),
    ) {
        val isSessionExpired: Boolean
            get() = session.expiresAt?.let { Instant.now().isAfter(it) } == true
    }

    fun expireSession(sessionId: TransferSessionId) {
        uploads[sessionId]?.let { upload ->
            upload.session = upload.session.copy(expiresAt = Instant.now().minusSeconds(60))
        }
    }

    private companion object {
        const val TRANSFER_CHUNK_BYTES = 64 * 1024
    }
}

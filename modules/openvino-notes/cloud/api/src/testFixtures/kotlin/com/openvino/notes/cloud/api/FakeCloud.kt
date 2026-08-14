// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package com.openvino.notes.cloud.api

import com.openvino.notes.kernel.AccountKey
import java.time.Instant

class FakeRemoteStore : RemoteObjectStore, RemoteChangeFeed, ResumableTransferClient {
    val objects = linkedMapOf<Pair<AccountKey, RemoteObjectId>, RemoteObject>()
    private val revisions = mutableMapOf<Pair<AccountKey, RemoteObjectId>, Long>()
    var changeOutcome: RemoteOutcome<RemoteChangePage> = RemoteOutcome.Success(RemoteChangePage(emptyList(), null))
    var startCursorOutcome: RemoteOutcome<RemoteCursor> = RemoteOutcome.Success(RemoteCursor("0"))

    override suspend fun list(accountKey: AccountKey): RemoteOutcome<List<RemoteObjectMetadata>> = RemoteOutcome.Success(
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

    override suspend fun get(accountKey: AccountKey, id: RemoteObjectId): RemoteOutcome<RemoteObject> =
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
        val objectValue = objects.remove(key)
            ?: return RemoteOutcome.Failure(RemoteError(RemoteErrorCode.NOT_FOUND, "fake.not_found"))
        val revision = (current ?: 0L) + 1L
        revisions[key] = revision
        return RemoteOutcome.Success(objectValue.metadata(RemoteRevision(revision.toString()), deleted = true))
    }

    override suspend fun startCursor(accountKey: AccountKey): RemoteOutcome<RemoteCursor> = startCursorOutcome

    override suspend fun changes(accountKey: AccountKey, cursor: RemoteCursor): RemoteOutcome<RemoteChangePage> = changeOutcome

    override suspend fun startUpload(
        accountKey: AccountKey,
        descriptor: UploadDescriptor,
        expectedRevision: RemoteRevision?,
    ): RemoteOutcome<UploadSession> = RemoteOutcome.Success(
        UploadSession(TransferSessionId("${accountKey.value}:${descriptor.objectId.value}"), descriptor.objectId, 0),
    )

    override suspend fun resumeUpload(
        accountKey: AccountKey,
        sessionId: TransferSessionId,
    ): RemoteOutcome<UploadSession> = RemoteOutcome.Failure(
        RemoteError(RemoteErrorCode.NOT_CONFIGURED, "fake.resume_not_configured"),
    )

    override suspend fun uploadChunk(
        accountKey: AccountKey,
        sessionId: TransferSessionId,
        chunk: UploadChunk,
    ): RemoteOutcome<UploadSession> = RemoteOutcome.Failure(
        RemoteError(RemoteErrorCode.NOT_CONFIGURED, "fake.upload_not_configured"),
    )

    override suspend fun downloadStream(
        accountKey: AccountKey,
        id: RemoteObjectId,
        sink: RemoteByteSink,
    ): RemoteOutcome<RemoteObjectMetadata> {
        val value = objects[accountKey to id]
            ?: return RemoteOutcome.Failure(RemoteError(RemoteErrorCode.NOT_FOUND, "fake.not_found"))
        sink.write(value.bytes)
        return RemoteOutcome.Success(value.metadata(RemoteRevision(revisions.getValue(accountKey to id).toString())))
    }

    private fun conflict(): RemoteOutcome.Failure =
        RemoteOutcome.Failure(RemoteError(RemoteErrorCode.CONFLICT, "fake.revision_conflict"))

    private fun RemoteObject.metadata(revision: RemoteRevision, deleted: Boolean = false) = RemoteObjectMetadata(
        id = id,
        name = name,
        mediaType = mediaType,
        modifiedAt = if (deleted) Instant.now() else modifiedAt,
        revision = revision,
        deleted = deleted,
    )
}

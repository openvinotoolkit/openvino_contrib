// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package com.openvino.notes.cloud.api

import com.openvino.notes.kernel.AccountKey
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class ResumableTransferContractTest {
    @Test fun `final upload chunk returns durable remote metadata and revision`() =
        runTest {
            val store = FakeRemoteStore()
            val accountKey = AccountKey("account")
            val objectId = RemoteObjectId("media")
            val session =
                store
                    .startUpload(
                        accountKey,
                        UploadDescriptor(objectId, "image.jpg", "image/jpeg", sizeBytes = 4),
                    ).value()
            assertFalse(session.toString().contains(session.id.value))

            val first =
                store
                    .uploadChunk(
                        accountKey,
                        session.id,
                        UploadChunk(offset = 0, bytes = byteArrayOf(1, 2), final = false),
                    ).value()
            assertEquals(2L, (first as UploadChunkOutcome.InProgress).session.nextOffset)

            val completed =
                store
                    .uploadChunk(
                        accountKey,
                        session.id,
                        UploadChunk(offset = 2, bytes = byteArrayOf(3, 4), final = true),
                    ).value() as UploadChunkOutcome.Completed

            assertEquals(objectId, completed.metadata.id)
            assertEquals(RemoteRevision("1"), completed.metadata.revision)
            assertArrayEquals(byteArrayOf(1, 2, 3, 4), store.get(accountKey, objectId).value().bytes)
        }

    @Test fun `resume upload returns active state with correct offset`() = runTest {
        val store = FakeRemoteStore()
        val accountKey = AccountKey("account")
        val session = store.startUpload(
            accountKey,
            UploadDescriptor(RemoteObjectId("media"), "image.jpg", "image/jpeg", sizeBytes = 4),
        ).value()

        store.uploadChunk(
            accountKey,
            session.id,
            UploadChunk(offset = 0, bytes = byteArrayOf(1, 2), final = false),
        )

        val state = store.resumeUpload(accountKey, session.id).value()
        val active = state as ResumableUploadState.Active
        assertEquals(2L, active.session.nextOffset)
    }

    @Test fun `resume upload returns completed state when upload already finished`() = runTest {
        val store = FakeRemoteStore()
        val accountKey = AccountKey("account")
        val objectId = RemoteObjectId("media")
        val session = store.startUpload(
            accountKey,
            UploadDescriptor(objectId, "image.jpg", "image/jpeg", sizeBytes = 4),
        ).value()

        store.uploadChunk(
            accountKey,
            session.id,
            UploadChunk(offset = 0, bytes = byteArrayOf(1, 2, 3, 4), final = true),
        )

        val state = store.resumeUpload(accountKey, session.id).value()
        val completed = state as ResumableUploadState.Completed
        assertEquals(objectId, completed.metadata.id)
        assertEquals(RemoteRevision("1"), completed.metadata.revision)
    }

    @Test fun `resume and chunk upload reject session expired`() = runTest {
        val store = FakeRemoteStore()
        val accountKey = AccountKey("account")
        val objectId = RemoteObjectId("media")
        val session = store.startUpload(
            accountKey,
            UploadDescriptor(objectId, "image.jpg", "image/jpeg", sizeBytes = 4),
        ).value()

        store.expireSession(session.id)

        val state = store.resumeUpload(accountKey, session.id).value()
        val expired = state as ResumableUploadState.Expired
        assertEquals(session.id, expired.session.id)

        val chunkOutcome = store.uploadChunk(
            accountKey,
            session.id,
            UploadChunk(offset = 0, bytes = byteArrayOf(1, 2), final = false),
        )
        assertTrue(chunkOutcome is RemoteOutcome.Failure)

        val getOutcome = store.get(accountKey, objectId)
        assertTrue(getOutcome is RemoteOutcome.Failure)
    }

    @Test fun `completed upload session retry returns cached metadata without overwriting later updates`() = runTest {
        val store = FakeRemoteStore()
        val accountKey = AccountKey("account")
        val objectId = RemoteObjectId("media")
        val session = store.startUpload(
            accountKey,
            UploadDescriptor(objectId,"image.jpg", "image/jpg", sizeBytes = 2)
        ).value()

        val chunk = UploadChunk(offset = 0, bytes = byteArrayOf(1, 2), final = true)
        val firstCompletion = store.uploadChunk(accountKey, session.id, chunk).value() as UploadChunkOutcome.Completed

        val session2 = store.startUpload(accountKey, UploadDescriptor(objectId, "image.jpg", "image/jpg", sizeBytes = 2)).value()
        val secondCompletion = store.uploadChunk(accountKey, session2.id,
            UploadChunk(offset = 0, bytes = byteArrayOf(3, 4), final = true)).value() as UploadChunkOutcome.Completed

        val retryOutcome = store.uploadChunk(accountKey, session.id, chunk).value() as UploadChunkOutcome.Completed

        assertEquals(firstCompletion.metadata.revision, retryOutcome.metadata.revision)

        assertArrayEquals(byteArrayOf(3, 4), store.get(accountKey, objectId).value().bytes)

        val remoteMetadata = store.list(accountKey).value().first { it.id == objectId }
        assertEquals(secondCompletion.metadata.revision, remoteMetadata.revision)
    }

    @Test fun `expired upload session cannot publish an object`() = runTest {
        val store = FakeRemoteStore()
        val accountKey = AccountKey("account")
        val objectId = RemoteObjectId("media")
        val session = store.startUpload(
            accountKey,
            UploadDescriptor(objectId, "image.jpg", "image/jpeg", sizeBytes = 2),
        ).value()

        store.expireSession(session.id)

        val outcome = store.uploadChunk(accountKey, session.id, UploadChunk(0, byteArrayOf(1, 2), final = true))

        assertTrue(outcome is RemoteOutcome.Failure)

        val getOutcome = store.get(accountKey, objectId)

        assertTrue(getOutcome is RemoteOutcome.Failure)
    }

    private fun <T> RemoteOutcome<T>.value(): T = (this as RemoteOutcome.Success<T>).value
}

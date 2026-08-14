// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package com.openvino.notes.cloud.api

import com.openvino.notes.kernel.AccountKey
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Test

class ResumableTransferContractTest {
    @Test fun `final upload chunk returns durable remote metadata and revision`() = runTest {
        val store = FakeRemoteStore()
        val accountKey = AccountKey("account")
        val objectId = RemoteObjectId("media")
        val session = store.startUpload(
            accountKey,
            UploadDescriptor(objectId, "image.jpg", "image/jpeg", sizeBytes = 4),
        ).value()

        val first = store.uploadChunk(
            accountKey,
            session.id,
            UploadChunk(offset = 0, bytes = byteArrayOf(1, 2), final = false),
        ).value()
        assertEquals(2L, (first as UploadChunkOutcome.InProgress).session.nextOffset)

        val completed = store.uploadChunk(
            accountKey,
            session.id,
            UploadChunk(offset = 2, bytes = byteArrayOf(3, 4), final = true),
        ).value() as UploadChunkOutcome.Completed

        assertEquals(objectId, completed.metadata.id)
        assertEquals(RemoteRevision("1"), completed.metadata.revision)
        assertArrayEquals(byteArrayOf(1, 2, 3, 4), store.get(accountKey, objectId).value().bytes)
    }

    private fun <T> RemoteOutcome<T>.value(): T = (this as RemoteOutcome.Success<T>).value
}

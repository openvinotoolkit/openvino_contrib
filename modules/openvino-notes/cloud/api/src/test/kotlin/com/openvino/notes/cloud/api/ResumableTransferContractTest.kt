// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package com.openvino.notes.cloud.api

import com.openvino.notes.kernel.AccountKey
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Test
import java.io.ByteArrayOutputStream
import java.time.Instant
import org.junit.Assert.assertTrue

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

    @Test fun `snapshot keeps revision consistent with streamed content`() =
        runTest {
            val store = FakeRemoteStore()
            val accountKey = AccountKey("account")
            val objectId = RemoteObjectId("media")
            store.put(
                accountKey,
                RemoteObject(objectId, "image.jpg", "image/jpeg", Instant.EPOCH, byteArrayOf(1, 2, 3)),
            ).value()

            val snapshot = store.getSnapshot(accountKey, objectId).value()
            val collected = ByteArrayOutputStream()
            snapshot.streamTo(RemoteByteSink { collected.write(it) })

            assertEquals(RemoteRevision("1"), snapshot.revision)
            assertArrayEquals(byteArrayOf(1, 2, 3), collected.toByteArray())
        }

    @Test fun `snapshot rejects concurrent modification as retryable conflict`() =
        runTest {
            val store = FakeRemoteStore()
            val accountKey = AccountKey("account")
            val objectId = RemoteObjectId("media")
            store.put(
                accountKey,
                RemoteObject(objectId, "image.jpg", "image/jpeg", Instant.EPOCH, byteArrayOf(1, 2, 3)),
            ).value()

            store.betweenMetadataAndContent = {
                store.put(
                    accountKey,
                    RemoteObject(objectId, "image.jpg", "image/jpeg", Instant.EPOCH, byteArrayOf(4, 5, 6)),
                ).value()
            }

            val outcome = store.getSnapshot(accountKey, objectId)
            assertEquals(RemoteErrorCode.CONFLICT, (outcome as RemoteOutcome.Failure).error.code)
        }

    @Test fun `snapshot streams content in chunks rather than one buffer`() =
        runTest {
            val store = FakeRemoteStore()
            val accountKey = AccountKey("account")
            val objectId = RemoteObjectId("media")
            val bigBytes = ByteArray(64 * 1024 * 2 + 100) { it.toByte() }
            store.put(
                accountKey,
                RemoteObject(objectId, "big.bin", "application/octet-stream", Instant.EPOCH, bigBytes),
            ).value()

            val snapshot = store.getSnapshot(accountKey, objectId).value()
            var writeCount = 0
            snapshot.streamTo(RemoteByteSink { writeCount++ })

            assertTrue("Expected multiple chunks, got $writeCount", writeCount > 1)
        }

    @Test fun `snapshot succeeds on retry after concurrent modification`() =
        runTest {
            val store = FakeRemoteStore()
            val accountKey = AccountKey("account")
            val objectId = RemoteObjectId("media")
            store.put(
                accountKey,
                RemoteObject(objectId, "image.jpg", "image/jpeg", Instant.EPOCH, byteArrayOf(1, 2, 3)),
            ).value()

            store.betweenMetadataAndContent = {
                store.put(
                    accountKey,
                    RemoteObject(objectId, "image.jpg", "image/jpeg", Instant.EPOCH, byteArrayOf(4, 5, 6)),
                ).value()
            }
            val first = store.getSnapshot(accountKey, objectId)
            assertEquals(RemoteErrorCode.CONFLICT, (first as RemoteOutcome.Failure).error.code)

            store.betweenMetadataAndContent = null
            val second = store.getSnapshot(accountKey, objectId).value()
            val collected = ByteArrayOutputStream()
            second.streamTo(RemoteByteSink { collected.write(it) })
            assertArrayEquals(byteArrayOf(4, 5, 6), collected.toByteArray())
        }
    private fun <T> RemoteOutcome<T>.value(): T = (this as RemoteOutcome.Success<T>).value
}

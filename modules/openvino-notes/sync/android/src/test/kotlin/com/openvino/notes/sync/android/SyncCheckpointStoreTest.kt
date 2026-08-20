// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package com.openvino.notes.sync.android

import android.content.Context
import androidx.room.Room
import androidx.test.core.app.ApplicationProvider
import com.openvino.notes.kernel.AccountKey
import com.openvino.notes.sync.api.port.OpaqueTransferSessionId
import com.openvino.notes.sync.api.port.PendingTransferCheckpoint
import com.openvino.notes.sync.api.port.SyncCheckpoint
import java.time.Instant
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [35])
class SyncCheckpointStoreTest {
    @Test fun `checkpoint round trip stays separate from consumer sync state`() = runTest {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val database = Room.inMemoryDatabaseBuilder(context, SyncDatabase::class.java).allowMainThreadQueries().build()
        try {
            val store = RoomSyncCheckpointStore(database.checkpointDao())
            val accountKey = AccountKey("account")
            val checkpoint = SyncCheckpoint(
                remoteCursor = "cursor",
                remoteRevisions = mapOf("note=1" to "revision\n2"),
                lastCompletedAt = Instant.EPOCH,
                resetRequired = true,
            )

            store.write(accountKey, checkpoint)

            assertEquals(checkpoint, store.read(accountKey))
            store.clear(accountKey)
            assertEquals(SyncCheckpoint(), store.read(accountKey))
        } finally {
            database.close()
        }
    }

    @Test fun `transfer checkpoint survives database reopen and stays account scoped`() = runTest {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val databaseName = "sync-transfer-checkpoint-test.db"
        context.deleteDatabase(databaseName)
        val accountKey = AccountKey("account")
        val otherAccountKey = AccountKey("other-account")
        val checkpoint = PendingTransferCheckpoint(
            operationId = "upload-attachment-1",
            objectKey = "attachment-1",
            sessionId = OpaqueTransferSessionId("sensitive-session-capability"),
            nextOffset = 70_000,
            expiresAt = Instant.parse("2026-08-16T00:00:00Z"),
        )
        try {
            val firstDatabase = Room.databaseBuilder(context, SyncDatabase::class.java, databaseName)
                .allowMainThreadQueries()
                .build()
            try {
                val store = RoomSyncTransferCheckpointStore(firstDatabase.transferCheckpointDao())
                store.write(accountKey, checkpoint)
                store.write(otherAccountKey, checkpoint.copy(operationId = "other-upload"))
            } finally {
                firstDatabase.close()
            }

            val reopenedDatabase = Room.databaseBuilder(context, SyncDatabase::class.java, databaseName)
                .allowMainThreadQueries()
                .build()
            try {
                val store = RoomSyncTransferCheckpointStore(reopenedDatabase.transferCheckpointDao())
                assertEquals(checkpoint, store.read(accountKey, checkpoint.operationId))
                assertEquals(listOf(checkpoint), store.pending(accountKey))
                assertEquals(false, checkpoint.toString().contains("sensitive-session-capability"))

                val resumed = checkpoint.copy(nextOffset = 96_000)
                store.write(accountKey, resumed)
                assertEquals(resumed, store.read(accountKey, checkpoint.operationId))
                assertEquals(true, store.remove(accountKey, checkpoint.operationId))
                assertEquals(emptyList<PendingTransferCheckpoint>(), store.pending(accountKey))
                assertEquals(1, store.pending(otherAccountKey).size)
                store.clear(otherAccountKey)
                assertEquals(emptyList<PendingTransferCheckpoint>(), store.pending(otherAccountKey))
            } finally {
                reopenedDatabase.close()
            }
        } finally {
            context.deleteDatabase(databaseName)
        }
    }
}

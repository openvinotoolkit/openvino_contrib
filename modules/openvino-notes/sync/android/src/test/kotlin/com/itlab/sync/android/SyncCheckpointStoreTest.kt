// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package com.itlab.sync.android

import android.content.Context
import androidx.room.Room
import androidx.test.core.app.ApplicationProvider
import com.itlab.kernel.AccountKey
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
}

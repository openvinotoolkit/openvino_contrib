// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package com.openvino.notes.sync.android

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import androidx.work.Data
import androidx.work.ListenableWorker
import androidx.work.testing.TestListenableWorkerBuilder
import com.openvino.notes.kernel.AccountKey
import com.openvino.notes.sync.api.SyncExecutor
import com.openvino.notes.sync.api.SyncOutcome
import com.openvino.notes.sync.api.SyncReason
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [35])
class SyncWorkerTest {
    @Test fun `worker maps retryable sync failure to retry`() = runBlocking {
        val context = ApplicationProvider.getApplicationContext<Context>()
        var executedAccount: AccountKey? = null
        var executedReason: SyncReason? = null
        val executor = SyncExecutor { accountKey, reason ->
            executedAccount = accountKey
            executedReason = reason
            SyncOutcome.Failed("remote.unavailable", retryable = true)
        }
        val worker = TestListenableWorkerBuilder<SyncWorker>(context)
            .setWorkerFactory(OpenVinoNotesWorkerFactory(executor))
            .setInputData(
                Data.Builder()
                    .putString("account_key", "account-a")
                    .putString("sync_reason", "USER")
                    .build(),
            )
            .build()

        assertEquals(ListenableWorker.Result.retry(), worker.doWork())
        assertEquals(AccountKey("account-a"), executedAccount)
        assertEquals(SyncReason.USER, executedReason)
    }

    @Test fun `account work identities and inputs are isolated`() {
        val accountA = AccountKey("account-a")
        val accountB = AccountKey("account-b")

        assertNotEquals(accountWorkTag(accountA), accountWorkTag(accountB))
        assertNotEquals(accountWorkName(accountA, "periodic"), accountWorkName(accountB, "periodic"))
        assertEquals("account-a", syncInputData(accountA, SyncReason.LOCAL_CHANGE).getString("account_key"))
        assertEquals("LOCAL_CHANGE", syncInputData(accountA, SyncReason.LOCAL_CHANGE).getString("sync_reason"))
    }

    @Test fun `worker rejects missing account identity`() = runBlocking {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val executor = SyncExecutor { _, _ -> SyncOutcome.Completed(0, 0) }
        val worker = TestListenableWorkerBuilder<SyncWorker>(context)
            .setWorkerFactory(OpenVinoNotesWorkerFactory(executor))
            .build()

        assertEquals(ListenableWorker.Result.failure(), worker.doWork())
    }
}

// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package com.itlab.sync.android

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import androidx.work.Data
import androidx.work.ListenableWorker
import androidx.work.testing.TestListenableWorkerBuilder
import com.itlab.sync.api.SyncExecutor
import com.itlab.sync.api.SyncOutcome
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [35])
class SyncWorkerTest {
    @Test fun `worker maps retryable sync failure to retry`() = runBlocking {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val executor = SyncExecutor { SyncOutcome.Failed("remote.unavailable", retryable = true) }
        val worker = TestListenableWorkerBuilder<SyncWorker>(context)
            .setWorkerFactory(OpenVinoNotesWorkerFactory(executor))
            .setInputData(Data.Builder().putString("sync_reason", "USER").build())
            .build()

        assertEquals(ListenableWorker.Result.retry(), worker.doWork())
    }
}

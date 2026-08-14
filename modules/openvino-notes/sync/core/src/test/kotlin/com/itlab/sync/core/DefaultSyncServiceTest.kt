// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package com.itlab.sync.core

import com.itlab.kernel.NoOpAppLogger
import com.itlab.sync.api.SyncBlockReason
import com.itlab.sync.api.SyncOutcome
import com.itlab.sync.api.SyncPhase
import com.itlab.sync.api.SyncReason
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Test

class DefaultSyncServiceTest {
    @Test fun `reports not configured without performing replication`() = runTest {
        val service = DisabledSyncService(NoOpAppLogger)

        val outcome = service.sync(SyncReason.USER)

        assertEquals(
            SyncOutcome.Blocked(SyncBlockReason.NotConfigured, "sync.not_configured"),
            outcome,
        )
        assertEquals(SyncPhase.BLOCKED, service.state.value.phase)
        assertEquals(SyncReason.USER, service.state.value.lastReason)
    }
}

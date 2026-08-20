// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package com.openvino.notes.sync.core

import com.openvino.notes.kernel.NoOpAppLogger
import com.openvino.notes.sync.api.SyncBlockReason
import com.openvino.notes.sync.api.SyncOutcome
import com.openvino.notes.sync.api.SyncPhase
import com.openvino.notes.sync.api.SyncReason
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

// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package com.itlab.sync.core

import com.itlab.kernel.AppLogger
import com.itlab.kernel.DiagnosticEvent
import com.itlab.kernel.DiagnosticLevel
import com.itlab.sync.api.SyncBlockReason
import com.itlab.sync.api.SyncExecutor
import com.itlab.sync.api.SyncOutcome
import com.itlab.sync.api.SyncPhase
import com.itlab.sync.api.SyncReason
import com.itlab.sync.api.SyncService
import com.itlab.sync.api.SyncState
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow

/**
 * Honest placeholder until a remote-first, revision-aware replication engine is implemented.
 * It deliberately performs no local or remote mutations.
 */
class DisabledSyncService(private val logger: AppLogger) : SyncService, SyncExecutor {
    private val mutableState = MutableStateFlow(
        SyncState(
            phase = SyncPhase.BLOCKED,
            diagnosticCode = NOT_CONFIGURED_CODE,
        ),
    )
    override val state: StateFlow<SyncState> = mutableState

    override suspend fun execute(reason: SyncReason): SyncOutcome = sync(reason)

    override suspend fun sync(reason: SyncReason): SyncOutcome {
        mutableState.value = SyncState(
            phase = SyncPhase.BLOCKED,
            lastReason = reason,
            diagnosticCode = NOT_CONFIGURED_CODE,
        )
        logger.log(
            DiagnosticEvent(
                code = NOT_CONFIGURED_CODE,
                level = DiagnosticLevel.INFO,
                attributes = mapOf("reason" to reason.name),
            ),
        )
        return SyncOutcome.Blocked(SyncBlockReason.NotConfigured, NOT_CONFIGURED_CODE)
    }

    private companion object {
        const val NOT_CONFIGURED_CODE = "sync.not_configured"
    }
}

data class SyncCoreComponent(val service: SyncService, val executor: SyncExecutor) {
    companion object {
        fun create(logger: AppLogger): SyncCoreComponent {
            val service = DisabledSyncService(logger)
            return SyncCoreComponent(service, service)
        }
    }
}

// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package com.openvino.notes.sync.core

import com.openvino.notes.kernel.AppLogger
import com.openvino.notes.kernel.AccountKey
import com.openvino.notes.kernel.DiagnosticEvent
import com.openvino.notes.kernel.DiagnosticLevel
import com.openvino.notes.sync.api.SyncBlockReason
import com.openvino.notes.sync.api.SyncOutcome
import com.openvino.notes.sync.api.SyncPhase
import com.openvino.notes.sync.api.SyncReason
import com.openvino.notes.sync.api.SyncService
import com.openvino.notes.sync.api.SyncState
import com.openvino.notes.sync.api.port.SyncCheckpointPort
import com.openvino.notes.sync.api.port.SyncExecutor
import com.openvino.notes.sync.api.port.SyncTransferCheckpointPort
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

    override suspend fun execute(accountKey: AccountKey, reason: SyncReason): SyncOutcome = sync(reason)

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

data class SyncCoreComponent(
    val service: SyncService,
    val executor: SyncExecutor,
    val checkpointPort: SyncCheckpointPort,
    val transferCheckpointPort: SyncTransferCheckpointPort,
) {
    companion object {
        fun create(
            logger: AppLogger,
            checkpointPort: SyncCheckpointPort,
            transferCheckpointPort: SyncTransferCheckpointPort,
        ): SyncCoreComponent {
            val service = DisabledSyncService(logger)
            return SyncCoreComponent(service, service, checkpointPort, transferCheckpointPort)
        }
    }
}

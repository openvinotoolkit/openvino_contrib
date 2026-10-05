// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package com.openvino.notes.sync.api

import java.time.Instant
import kotlinx.coroutines.flow.StateFlow

enum class SyncReason { USER, STARTUP, PERIODIC, LOCAL_CHANGE }
enum class SyncPhase { IDLE, RUNNING, BLOCKED, FAILED }

data class SyncState(
    val phase: SyncPhase = SyncPhase.IDLE,
    val lastSuccessfulAt: Instant? = null,
    val lastReason: SyncReason? = null,
    val diagnosticCode: String? = null,
)

sealed interface SyncBlockReason {
    data object NotConfigured : SyncBlockReason
    data object SignedOut : SyncBlockReason
    data object NotAuthorized : SyncBlockReason
    data object RemoteResetRequired : SyncBlockReason
}

sealed interface SyncOutcome {
    data class Completed(val uploaded: Int, val downloaded: Int) : SyncOutcome
    data object SignedOut : SyncOutcome
    data class Blocked(val reason: SyncBlockReason, val code: String) : SyncOutcome
    data class Failed(val code: String, val retryable: Boolean) : SyncOutcome
}

interface SyncService {
    val state: StateFlow<SyncState>
    suspend fun sync(reason: SyncReason): SyncOutcome
}

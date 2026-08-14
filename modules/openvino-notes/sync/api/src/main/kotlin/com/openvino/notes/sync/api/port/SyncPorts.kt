// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package com.openvino.notes.sync.api.port

import com.openvino.notes.kernel.AccountKey
import com.openvino.notes.sync.api.SyncOutcome
import com.openvino.notes.sync.api.SyncReason
import java.time.Instant

fun interface SyncExecutor {
    suspend fun execute(accountKey: AccountKey, reason: SyncReason): SyncOutcome
}

interface SyncScheduler {
    fun schedulePeriodic(accountKey: AccountKey)
    fun request(accountKey: AccountKey, reason: SyncReason)
    fun cancel(accountKey: AccountKey)
}

/** Internal replication state. It is intentionally separate from consumer-facing SyncState. */
data class SyncCheckpoint(
    val remoteCursor: String? = null,
    val remoteRevisions: Map<String, String> = emptyMap(),
    val lastCompletedAt: Instant? = null,
    val resetRequired: Boolean = false,
)

interface SyncCheckpointPort {
    suspend fun read(accountKey: AccountKey): SyncCheckpoint
    suspend fun write(accountKey: AccountKey, checkpoint: SyncCheckpoint)
    suspend fun clear(accountKey: AccountKey)
}

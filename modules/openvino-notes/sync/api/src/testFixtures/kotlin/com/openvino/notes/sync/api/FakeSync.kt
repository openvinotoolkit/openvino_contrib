// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package com.openvino.notes.sync.api

import com.openvino.notes.kernel.AccountKey
import com.openvino.notes.sync.api.port.PendingTransferCheckpoint
import com.openvino.notes.sync.api.port.SyncCheckpoint
import com.openvino.notes.sync.api.port.SyncCheckpointPort
import com.openvino.notes.sync.api.port.SyncTransferCheckpointPort
import kotlinx.coroutines.flow.MutableStateFlow

class FakeSyncService(initial: SyncState = SyncState()) : SyncService {
    override val state = MutableStateFlow(initial)
    var outcome: SyncOutcome = SyncOutcome.Completed(0, 0)
    override suspend fun sync(reason: SyncReason): SyncOutcome = outcome
}

class FakeSyncCheckpointPort : SyncCheckpointPort {
    private val checkpoints = mutableMapOf<AccountKey, SyncCheckpoint>()
    override suspend fun read(accountKey: AccountKey): SyncCheckpoint = checkpoints[accountKey] ?: SyncCheckpoint()
    override suspend fun write(accountKey: AccountKey, checkpoint: SyncCheckpoint) {
        checkpoints[accountKey] = checkpoint
    }
    override suspend fun clear(accountKey: AccountKey) {
        checkpoints.remove(accountKey)
    }
}

class FakeSyncTransferCheckpointPort : SyncTransferCheckpointPort {
    private val checkpoints = mutableMapOf<Pair<AccountKey, String>, PendingTransferCheckpoint>()
    override suspend fun read(accountKey: AccountKey, operationId: String): PendingTransferCheckpoint? =
        checkpoints[accountKey to operationId]
    override suspend fun pending(accountKey: AccountKey): List<PendingTransferCheckpoint> =
        checkpoints.filterKeys { (key, _) -> key == accountKey }.values.sortedBy(PendingTransferCheckpoint::operationId)
    override suspend fun write(accountKey: AccountKey, checkpoint: PendingTransferCheckpoint) {
        checkpoints[accountKey to checkpoint.operationId] = checkpoint
    }
    override suspend fun remove(accountKey: AccountKey, operationId: String): Boolean =
        checkpoints.remove(accountKey to operationId) != null
    override suspend fun clear(accountKey: AccountKey) {
        checkpoints.keys.removeAll { (key, _) -> key == accountKey }
    }
}

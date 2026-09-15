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

/** Sensitive Cloud session value persisted opaquely by Sync without depending on Cloud API types. */
class OpaqueTransferSessionId(val value: String) {
    init { require(value.isNotBlank()) { "Transfer session ID must not be blank" } }

    override fun equals(other: Any?): Boolean = other is OpaqueTransferSessionId && value == other.value
    override fun hashCode(): Int = value.hashCode()
    override fun toString(): String = "OpaqueTransferSessionId(**redacted**)"
}

data class PendingTransferCheckpoint(
    val operationId: String,
    val objectKey: String,
    val sessionId: OpaqueTransferSessionId,
    val nextOffset: Long,
    val expiresAt: Instant? = null,
) {
    init {
        require(operationId.isNotBlank()) { "Transfer operation ID must not be blank" }
        require(objectKey.isNotBlank()) { "Transfer object key must not be blank" }
        require(nextOffset >= 0) { "Transfer offset must not be negative" }
    }
}

interface SyncCheckpointPort {
    suspend fun read(accountKey: AccountKey): SyncCheckpoint
    suspend fun write(accountKey: AccountKey, checkpoint: SyncCheckpoint)
    suspend fun clear(accountKey: AccountKey)
}

/** Durable, account-scoped state required to resume media transfers after process death. */
interface SyncTransferCheckpointPort {
    suspend fun read(accountKey: AccountKey, operationId: String): PendingTransferCheckpoint?
    suspend fun pending(accountKey: AccountKey): List<PendingTransferCheckpoint>
    suspend fun write(accountKey: AccountKey, checkpoint: PendingTransferCheckpoint)
    suspend fun remove(accountKey: AccountKey, operationId: String): Boolean
    suspend fun clear(accountKey: AccountKey)
}

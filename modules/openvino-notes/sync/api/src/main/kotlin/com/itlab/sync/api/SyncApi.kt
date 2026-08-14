package com.itlab.sync.api

import com.itlab.identity.api.AccountId
import java.time.Instant
import kotlinx.coroutines.flow.StateFlow

enum class SyncReason { USER, STARTUP, PERIODIC, LOCAL_CHANGE }
enum class SyncPhase { IDLE, RUNNING, BLOCKED, FAILED }

data class SyncState(
    val phase: SyncPhase = SyncPhase.IDLE,
    val lastSuccessfulAt: Instant? = null,
    val diagnosticCode: String? = null,
    val continuationToken: String? = null,
)

sealed interface SyncOutcome {
    data class Completed(val uploaded: Int, val downloaded: Int) : SyncOutcome
    data object SignedOut : SyncOutcome
    data class Blocked(val code: String) : SyncOutcome
    data class Failed(val code: String, val retryable: Boolean) : SyncOutcome
}

interface SyncService {
    val state: StateFlow<SyncState>
    suspend fun sync(reason: SyncReason): SyncOutcome
}

fun interface SyncExecutor {
    suspend fun execute(reason: SyncReason): SyncOutcome
}

interface SyncScheduler {
    fun schedulePeriodic()
    fun request(reason: SyncReason)
    fun cancelAll()
}

interface SyncStateStore {
    suspend fun read(accountId: AccountId): SyncState
    suspend fun write(accountId: AccountId, state: SyncState)
}

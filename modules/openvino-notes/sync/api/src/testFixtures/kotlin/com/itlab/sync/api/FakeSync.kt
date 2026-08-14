package com.itlab.sync.api

import com.itlab.identity.api.AccountId
import kotlinx.coroutines.flow.MutableStateFlow

class FakeSyncService(initial: SyncState = SyncState()) : SyncService {
    override val state = MutableStateFlow(initial)
    var outcome: SyncOutcome = SyncOutcome.Completed(0, 0)
    override suspend fun sync(reason: SyncReason): SyncOutcome = outcome
}

class FakeSyncStateStore : SyncStateStore {
    private val values = mutableMapOf<AccountId, SyncState>()
    override suspend fun read(accountId: AccountId): SyncState = values[accountId] ?: SyncState()
    override suspend fun write(accountId: AccountId, state: SyncState) { values[accountId] = state }
}

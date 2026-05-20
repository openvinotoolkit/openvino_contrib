package com.itlab.domain.cloud

import kotlinx.coroutines.flow.StateFlow

interface SyncManager {
    val syncState: StateFlow<SyncState>

    /** First-time (or forced) full push + pull. */
    suspend fun syncFull(userId: String)

    /** Push local deltas (if any), then pull remote deltas. */
    suspend fun syncIncremental(userId: String)

    /** Upload only unsynced notes / new media files — no pull. */
    suspend fun pushLocalChanges(userId: String)

    suspend fun hasPendingLocalChanges(userId: String): Boolean

    suspend fun hasPendingLocalChangesForNote(
        userId: String,
        noteId: String,
    ): Boolean

    suspend fun sync(userId: String)

    suspend fun pushChanges(userId: String)

    suspend fun pullUpdates(userId: String)
}

sealed class SyncState {
    object Idle : SyncState()

    object Syncing : SyncState()

    data class Error(
        val message: String,
    ) : SyncState()

    object Success : SyncState()
}

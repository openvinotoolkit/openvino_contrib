package com.itlab.domain.cloud

/** Tracks per-user one-time full cloud sync on this device. */
interface SyncCheckpointStore {
    suspend fun hasCompletedInitialFullSync(userId: String): Boolean

    suspend fun markInitialFullSyncCompleted(userId: String)

    suspend fun clearUser(userId: String)
}

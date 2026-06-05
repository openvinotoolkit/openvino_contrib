package com.itlab.data.cloud

import android.content.Context
import androidx.core.content.edit
import com.itlab.domain.cloud.SyncCheckpointStore

class SharedPreferencesSyncCheckpointStore(
    context: Context,
) : SyncCheckpointStore {
    private val prefs =
        context.applicationContext.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

    override suspend fun hasCompletedInitialFullSync(userId: String): Boolean =
        prefs.getStringSet(KEY_COMPLETED_USER_IDS, emptySet())?.contains(userId) == true

    override suspend fun markInitialFullSyncCompleted(userId: String) {
        val current = prefs.getStringSet(KEY_COMPLETED_USER_IDS, emptySet())?.toMutableSet() ?: mutableSetOf()
        current.add(userId)
        prefs.edit {
            putStringSet(KEY_COMPLETED_USER_IDS, current)
        }
    }

    override suspend fun clearUser(userId: String) {
        val current = prefs.getStringSet(KEY_COMPLETED_USER_IDS, emptySet())?.toMutableSet() ?: mutableSetOf()
        current.remove(userId)
        prefs.edit {
            putStringSet(KEY_COMPLETED_USER_IDS, current)
        }
    }

    private companion object {
        const val PREFS_NAME = "sync_checkpoint"
        const val KEY_COMPLETED_USER_IDS = "initial_full_sync_user_ids"
    }
}

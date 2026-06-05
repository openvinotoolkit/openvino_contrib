package com.itlab.data.cloud

import android.content.Context
import androidx.work.CoroutineWorker
import androidx.work.WorkerParameters
import com.itlab.data.cloud.AuthManager
import com.itlab.domain.cloud.SyncCheckpointStore
import com.itlab.domain.cloud.SyncManager
import kotlinx.coroutines.CancellationException
import timber.log.Timber

class SyncWorker(
    context: Context,
    params: WorkerParameters,
    private val syncManager: SyncManager,
    private val authManager: AuthManager,
    private val syncCheckpointStore: SyncCheckpointStore,
) : CoroutineWorker(context, params) {
    @Suppress("TooGenericExceptionCaught", "ReturnCount")
    override suspend fun doWork(): Result {
        val userId =
            inputData.getString("USER_ID")
                ?: authManager.getCurrentUserId()
                ?: run {
                    Timber.e("Sync failed: User is not authorized")
                    return Result.failure()
                }

        val isTokenValid = authManager.refreshAuthToken()
        if (!isTokenValid) {
            Timber.w("Sync deferred: Firebase token is temporary unavailable. Retrying...")
            return Result.retry()
        }

        return try {
            val needsFullSync = !syncCheckpointStore.hasCompletedInitialFullSync(userId)
            if (needsFullSync) {
                Timber.d("Starting initial full sync for user: $userId")
                syncManager.syncFull(userId)
                syncCheckpointStore.markInitialFullSyncCompleted(userId)
            } else if (syncManager.hasPendingLocalChanges(userId)) {
                Timber.d("Starting incremental sync (local changes) for user: $userId")
                syncManager.syncIncremental(userId)
            } else {
                Timber.d("Skipping background sync — no local changes for user: $userId")
            }

            Timber.d("Sync completed successfully")
            Result.success()
        } catch (e: CancellationException) {
            throw e
        } catch (e: java.io.IOException) {
            Timber.e(e, "Sync retryable error: %s", e.message)
            Result.retry()
        } catch (e: com.google.firebase.FirebaseException) {
            Timber.e(e, "Sync retryable Firebase/Storage error: %s", e.message)
            Result.retry()
        } catch (e: Exception) {
            Timber.e(e, "Sync fatal error: %s", e.message)
            Result.failure()
        }
    }
}

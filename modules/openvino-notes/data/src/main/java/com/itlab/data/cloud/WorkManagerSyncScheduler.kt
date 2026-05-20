package com.itlab.data.cloud

import androidx.work.BackoffPolicy
import androidx.work.Constraints
import androidx.work.ExistingWorkPolicy
import androidx.work.NetworkType
import androidx.work.OneTimeWorkRequestBuilder
import androidx.work.WorkManager
import androidx.work.workDataOf
import com.itlab.data.cloud.SyncWorker
import com.itlab.domain.cloud.SyncScheduler
import java.util.concurrent.TimeUnit

class WorkManagerSyncScheduler(
    private val workManager: WorkManager,
) : SyncScheduler {
    override fun scheduleSync(userId: String) {
        val constraints =
            Constraints
                .Builder()
                .setRequiredNetworkType(NetworkType.CONNECTED)
                .build()

        val syncRequest =
            OneTimeWorkRequestBuilder<SyncWorker>()
                .setConstraints(constraints)
                .setInputData(workDataOf("USER_ID" to userId))
                .setBackoffCriteria(
                    BackoffPolicy.EXPONENTIAL,
                    10,
                    TimeUnit.SECONDS,
                ).build()

        workManager.enqueueUniqueWork(
            "sync_work_$userId",
            ExistingWorkPolicy.KEEP,
            syncRequest,
        )
    }
}

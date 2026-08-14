// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package com.itlab.sync.android

import android.content.Context
import androidx.room.Dao
import androidx.room.Database
import androidx.room.Entity
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import androidx.room.Room
import androidx.room.RoomDatabase
import androidx.work.CoroutineWorker
import androidx.work.Data
import androidx.work.ExistingPeriodicWorkPolicy
import androidx.work.ExistingWorkPolicy
import androidx.work.ListenableWorker
import androidx.work.OneTimeWorkRequestBuilder
import androidx.work.PeriodicWorkRequestBuilder
import androidx.work.WorkManager
import androidx.work.WorkerFactory
import androidx.work.WorkerParameters
import com.itlab.kernel.AccountKey
import com.itlab.sync.api.SyncExecutor
import com.itlab.sync.api.SyncOutcome
import com.itlab.sync.api.SyncReason
import com.itlab.sync.api.SyncScheduler
import java.time.Instant
import java.util.concurrent.TimeUnit

internal data class SyncCheckpoint(
    val remoteCursor: String? = null,
    val remoteRevisions: Map<String, String> = emptyMap(),
    val lastCompletedAt: Instant? = null,
    val resetRequired: Boolean = false,
)

internal interface SyncCheckpointStore {
    suspend fun read(accountKey: AccountKey): SyncCheckpoint
    suspend fun write(accountKey: AccountKey, checkpoint: SyncCheckpoint)
    suspend fun clear(accountKey: AccountKey)
}

internal class SyncWorker(
    appContext: Context,
    params: WorkerParameters,
    private val executor: SyncExecutor,
) : CoroutineWorker(appContext, params) {
    override suspend fun doWork(): Result {
        val reason = inputData.getString(KEY_REASON)?.let { runCatching { SyncReason.valueOf(it) }.getOrNull() } ?: SyncReason.PERIODIC
        return when (val outcome = executor.execute(reason)) {
            is SyncOutcome.Completed, SyncOutcome.SignedOut, is SyncOutcome.Blocked -> Result.success()
            is SyncOutcome.Failed -> if (outcome.retryable) Result.retry() else Result.failure()
        }
    }
}

class OpenVinoNotesWorkerFactory(private val executor: SyncExecutor) : WorkerFactory() {
    override fun createWorker(appContext: Context, workerClassName: String, workerParameters: WorkerParameters): ListenableWorker? =
        if (workerClassName == SyncWorker::class.java.name) SyncWorker(appContext, workerParameters, executor) else null
}

internal class WorkManagerSyncScheduler(private val workManager: WorkManager) : SyncScheduler {
    override fun schedulePeriodic() {
        val request = PeriodicWorkRequestBuilder<SyncWorker>(15, TimeUnit.MINUTES)
            .setInputData(reasonData(SyncReason.PERIODIC))
            .addTag(TAG)
            .build()
        workManager.enqueueUniquePeriodicWork(PERIODIC_WORK, ExistingPeriodicWorkPolicy.UPDATE, request)
    }

    override fun request(reason: SyncReason) {
        val request = OneTimeWorkRequestBuilder<SyncWorker>().setInputData(reasonData(reason)).addTag(TAG).build()
        workManager.enqueueUniqueWork("$ONE_TIME_WORK.${reason.name.lowercase()}", ExistingWorkPolicy.REPLACE, request)
    }

    override fun cancelAll() { workManager.cancelAllWorkByTag(TAG) }

    private fun reasonData(reason: SyncReason): Data = Data.Builder().putString(KEY_REASON, reason.name).build()
}

@Entity(tableName = "sync_checkpoint")
internal data class SyncCheckpointEntity(
    @androidx.room.PrimaryKey val accountKey: String,
    val remoteCursor: String?,
    val remoteRevisions: String,
    val lastCompletedAtMillis: Long?,
    val resetRequired: Boolean,
)

@Dao
internal interface SyncCheckpointDao {
    @Query("SELECT * FROM sync_checkpoint WHERE accountKey = :accountKey")
    suspend fun read(accountKey: String): SyncCheckpointEntity?

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun write(checkpoint: SyncCheckpointEntity)

    @Query("DELETE FROM sync_checkpoint WHERE accountKey = :accountKey")
    suspend fun clear(accountKey: String)
}

@Database(entities = [SyncCheckpointEntity::class], version = 1, exportSchema = true)
internal abstract class SyncDatabase : RoomDatabase() {
    abstract fun checkpointDao(): SyncCheckpointDao
}

internal class RoomSyncCheckpointStore(private val dao: SyncCheckpointDao) : SyncCheckpointStore {
    override suspend fun read(accountKey: AccountKey): SyncCheckpoint =
        dao.read(accountKey.value)?.toApi() ?: SyncCheckpoint()

    override suspend fun write(accountKey: AccountKey, checkpoint: SyncCheckpoint) {
        dao.write(checkpoint.toEntity(accountKey))
    }

    override suspend fun clear(accountKey: AccountKey) {
        dao.clear(accountKey.value)
    }
}

internal class AndroidSyncCheckpointComponent(
    private val database: SyncDatabase,
    val checkpointStore: SyncCheckpointStore,
) : AutoCloseable {
    override fun close() = database.close()
}

object AndroidSyncComponent {
    internal fun openCheckpointStore(context: Context): AndroidSyncCheckpointComponent {
        val database = Room.databaseBuilder(context.applicationContext, SyncDatabase::class.java, "openvino-notes-sync.db").build()
        return AndroidSyncCheckpointComponent(database, RoomSyncCheckpointStore(database.checkpointDao()))
    }

    fun createWorkerFactory(executor: SyncExecutor): WorkerFactory = OpenVinoNotesWorkerFactory(executor)

    fun createScheduler(context: Context): SyncScheduler = WorkManagerSyncScheduler(WorkManager.getInstance(context.applicationContext))
}

private fun SyncCheckpointEntity.toApi() = SyncCheckpoint(
    remoteCursor = remoteCursor,
    remoteRevisions = decodeRevisions(remoteRevisions),
    lastCompletedAt = lastCompletedAtMillis?.let(Instant::ofEpochMilli),
    resetRequired = resetRequired,
)

private fun SyncCheckpoint.toEntity(accountKey: AccountKey) = SyncCheckpointEntity(
    accountKey = accountKey.value,
    remoteCursor = remoteCursor,
    remoteRevisions = remoteRevisions.entries
        .sortedBy(Map.Entry<String, String>::key)
        .joinToString("\n") { "${escape(it.key)}=${escape(it.value)}" },
    lastCompletedAtMillis = lastCompletedAt?.toEpochMilli(),
    resetRequired = resetRequired,
)

private fun decodeRevisions(value: String): Map<String, String> = value.lineSequence()
    .filter(String::isNotEmpty)
    .associate { line ->
        val separator = line.indexOf('=')
        require(separator >= 0) { "Invalid sync revision entry" }
        unescape(line.substring(0, separator)) to unescape(line.substring(separator + 1))
    }

private fun escape(value: String): String = java.net.URLEncoder.encode(value, Charsets.UTF_8.name())
private fun unescape(value: String): String = java.net.URLDecoder.decode(value, Charsets.UTF_8.name())

private const val KEY_REASON = "sync_reason"
private const val PERIODIC_WORK = "openvino-notes.sync.periodic"
private const val ONE_TIME_WORK = "openvino-notes.sync.once"
private const val TAG = "openvino-notes.sync"

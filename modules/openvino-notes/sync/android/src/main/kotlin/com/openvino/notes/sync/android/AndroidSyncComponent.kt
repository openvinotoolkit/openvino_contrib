// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package com.openvino.notes.sync.android

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
import com.openvino.notes.kernel.AccountKey
import com.openvino.notes.sync.api.SyncExecutor
import com.openvino.notes.sync.api.SyncCheckpoint
import com.openvino.notes.sync.api.SyncCheckpointPort
import com.openvino.notes.sync.api.SyncOutcome
import com.openvino.notes.sync.api.SyncReason
import com.openvino.notes.sync.api.SyncScheduler
import java.security.MessageDigest
import java.time.Instant
import java.util.concurrent.TimeUnit

internal class SyncWorker(
    appContext: Context,
    params: WorkerParameters,
    private val executor: SyncExecutor,
) : CoroutineWorker(appContext, params) {
    override suspend fun doWork(): Result {
        val accountKey = inputData.getString(KEY_ACCOUNT_KEY)?.takeIf(String::isNotBlank)?.let(::AccountKey)
            ?: return Result.failure()
        val reason = inputData.getString(KEY_REASON)?.let { runCatching { SyncReason.valueOf(it) }.getOrNull() } ?: SyncReason.PERIODIC
        return when (val outcome = executor.execute(accountKey, reason)) {
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
    override fun schedulePeriodic(accountKey: AccountKey) {
        val request = PeriodicWorkRequestBuilder<SyncWorker>(15, TimeUnit.MINUTES)
            .setInputData(syncInputData(accountKey, SyncReason.PERIODIC))
            .addTag(accountWorkTag(accountKey))
            .build()
        workManager.enqueueUniquePeriodicWork(
            accountWorkName(accountKey, "periodic"),
            ExistingPeriodicWorkPolicy.UPDATE,
            request,
        )
    }

    override fun request(accountKey: AccountKey, reason: SyncReason) {
        val request = OneTimeWorkRequestBuilder<SyncWorker>()
            .setInputData(syncInputData(accountKey, reason))
            .addTag(accountWorkTag(accountKey))
            .build()
        workManager.enqueueUniqueWork(
            accountWorkName(accountKey, "once.${reason.name.lowercase()}"),
            ExistingWorkPolicy.REPLACE,
            request,
        )
    }

    override fun cancel(accountKey: AccountKey) {
        workManager.cancelAllWorkByTag(accountWorkTag(accountKey))
    }
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

internal class RoomSyncCheckpointStore(private val dao: SyncCheckpointDao) : SyncCheckpointPort {
    override suspend fun read(accountKey: AccountKey): SyncCheckpoint =
        dao.read(accountKey.value)?.toApi() ?: SyncCheckpoint()

    override suspend fun write(accountKey: AccountKey, checkpoint: SyncCheckpoint) {
        dao.write(checkpoint.toEntity(accountKey))
    }

    override suspend fun clear(accountKey: AccountKey) {
        dao.clear(accountKey.value)
    }
}

class AndroidSyncCheckpointComponent internal constructor(
    private val database: SyncDatabase,
    val checkpointPort: SyncCheckpointPort,
) : AutoCloseable {
    override fun close() = database.close()
}

object AndroidSyncComponent {
    fun openCheckpointStore(context: Context): AndroidSyncCheckpointComponent {
        val database = Room.databaseBuilder(context.applicationContext, SyncDatabase::class.java, "openvino-notes-sync.db").build()
        return AndroidSyncCheckpointComponent(database, RoomSyncCheckpointStore(database.checkpointDao()))
    }

    fun createWorkerFactory(executor: SyncExecutor): WorkerFactory = OpenVinoNotesWorkerFactory(executor)

    fun createScheduler(context: Context): SyncScheduler = WorkManagerSyncScheduler(WorkManager.getInstance(context.applicationContext))
}

internal fun syncInputData(accountKey: AccountKey, reason: SyncReason): Data = Data.Builder()
    .putString(KEY_ACCOUNT_KEY, accountKey.value)
    .putString(KEY_REASON, reason.name)
    .build()

internal fun accountWorkTag(accountKey: AccountKey): String = "openvino-notes.sync.${accountScope(accountKey)}"

internal fun accountWorkName(accountKey: AccountKey, suffix: String): String = "${accountWorkTag(accountKey)}.$suffix"

private fun accountScope(accountKey: AccountKey): String = MessageDigest.getInstance("SHA-256")
    .digest(accountKey.value.toByteArray(Charsets.UTF_8))
    .take(12)
    .joinToString("") { byte -> "%02x".format(byte) }

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
private const val KEY_ACCOUNT_KEY = "account_key"

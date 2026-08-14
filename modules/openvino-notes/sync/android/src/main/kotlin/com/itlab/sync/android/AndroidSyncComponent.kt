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
import com.itlab.identity.api.AccountId
import com.itlab.sync.api.SyncExecutor
import com.itlab.sync.api.SyncOutcome
import com.itlab.sync.api.SyncPhase
import com.itlab.sync.api.SyncReason
import com.itlab.sync.api.SyncScheduler
import com.itlab.sync.api.SyncState
import com.itlab.sync.api.SyncStateStore
import java.time.Instant
import java.util.concurrent.TimeUnit

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

@Entity(tableName = "sync_state")
internal data class SyncStateEntity(
    @androidx.room.PrimaryKey val accountId: String,
    val phase: String,
    val lastSuccessfulAtMillis: Long?,
    val diagnosticCode: String?,
    val continuationToken: String?,
)

@Dao
internal interface SyncStateDao {
    @Query("SELECT * FROM sync_state WHERE accountId = :accountId") suspend fun read(accountId: String): SyncStateEntity?
    @Insert(onConflict = OnConflictStrategy.REPLACE) suspend fun write(state: SyncStateEntity)
}

@Database(entities = [SyncStateEntity::class], version = 1, exportSchema = true)
internal abstract class SyncDatabase : RoomDatabase() {
    abstract fun stateDao(): SyncStateDao
}

internal class RoomSyncStateStore(private val dao: SyncStateDao) : SyncStateStore {
    override suspend fun read(accountId: AccountId): SyncState = dao.read(accountId.value)?.toApi() ?: SyncState()
    override suspend fun write(accountId: AccountId, state: SyncState) { dao.write(state.toEntity(accountId)) }
}

class AndroidSyncStateComponent internal constructor(
    private val database: SyncDatabase,
    val stateStore: SyncStateStore,
) : AutoCloseable {
    override fun close() = database.close()
}

data class AndroidSyncRuntimeComponent(
    val scheduler: SyncScheduler,
    val workerFactory: WorkerFactory,
)

object AndroidSyncComponent {
    fun openStateStore(context: Context): AndroidSyncStateComponent {
        val database = Room.databaseBuilder(context.applicationContext, SyncDatabase::class.java, "openvino-notes-sync.db").build()
        return AndroidSyncStateComponent(database, RoomSyncStateStore(database.stateDao()))
    }

    fun createRuntime(context: Context, executor: SyncExecutor): AndroidSyncRuntimeComponent {
        val appContext = context.applicationContext
        return AndroidSyncRuntimeComponent(
            WorkManagerSyncScheduler(WorkManager.getInstance(appContext)),
            OpenVinoNotesWorkerFactory(executor),
        )
    }
}

private fun SyncStateEntity.toApi() = SyncState(SyncPhase.valueOf(phase), lastSuccessfulAtMillis?.let(Instant::ofEpochMilli), diagnosticCode, continuationToken)
private fun SyncState.toEntity(accountId: AccountId) = SyncStateEntity(accountId.value, phase.name, lastSuccessfulAt?.toEpochMilli(), diagnosticCode, continuationToken)

private const val KEY_REASON = "sync_reason"
private const val PERIODIC_WORK = "openvino-notes.sync.periodic"
private const val ONE_TIME_WORK = "openvino-notes.sync.once"
private const val TAG = "openvino-notes.sync"

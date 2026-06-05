package com.itlab.data.cloud

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import androidx.work.ListenableWorker.Result
import androidx.work.testing.TestListenableWorkerBuilder
import com.itlab.domain.cloud.SyncCheckpointStore
import com.itlab.domain.cloud.SyncManager
import io.mockk.Runs
import io.mockk.clearAllMocks
import io.mockk.coEvery
import io.mockk.every
import io.mockk.just
import io.mockk.mockk
import kotlinx.coroutines.runBlocking
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import timber.log.Timber
import java.io.IOException

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class SyncWorkerTest {
    private lateinit var context: Context
    private val syncManager = mockk<SyncManager>()
    private val authManager = mockk<AuthManager>()
    private val syncCheckpointStore = mockk<SyncCheckpointStore>()

    private fun stubIncrementalSyncReady() {
        coEvery { syncCheckpointStore.hasCompletedInitialFullSync(any()) } returns true
        coEvery { syncCheckpointStore.markInitialFullSyncCompleted(any()) } just Runs
        coEvery { syncManager.hasPendingLocalChanges(any()) } returns true
        coEvery { syncManager.hasPendingLocalChangesForNote(any(), any()) } returns false
        coEvery { syncManager.syncIncremental(any()) } just Runs
    }

    private fun createWorker() =
        TestListenableWorkerBuilder<SyncWorker>(context)
            .setWorkerFactory(
                object : androidx.work.WorkerFactory() {
                    override fun createWorker(
                        appContext: Context,
                        workerClassName: String,
                        workerParameters: androidx.work.WorkerParameters,
                    ) = SyncWorker(appContext, workerParameters, syncManager, authManager, syncCheckpointStore)
                },
            ).build()

    @Before
    fun setUp() {
        context = ApplicationProvider.getApplicationContext()

        Timber.plant(
            object : Timber.Tree() {
                override fun log(
                    priority: Int,
                    tag: String?,
                    message: String,
                    t: Throwable?,
                ) {
                    // Пакетная заглушка для тестов: логи не нужны
                }
            },
        )
    }

    @After
    fun tearDown() {
        Timber.uprootAll()
        clearAllMocks()
    }

    @Test
    fun `doWork should return success when sync is successful`() =
        runBlocking {
            val userId = "user_1"
            every { authManager.getCurrentUserId() } returns userId
            // ФИКС: Мокаем успешное обновление токена
            coEvery { authManager.refreshAuthToken() } returns true
            stubIncrementalSyncReady()

            val result = createWorker().doWork()

            assertEquals(Result.success(), result)
        }

    @Test
    fun `doWork should return failure when user is not authorized`() =
        runBlocking {
            every { authManager.getCurrentUserId() } returns null

            val result = createWorker().doWork()

            assertEquals(Result.failure(), result)
        }

    // НОВЫЙ ТЕСТ: Проверяем, что если токен временно недоступен, воркер уходит в RETRY
    @Test
    fun `doWork should return retry when auth token refresh fails`() =
        runBlocking {
            val userId = "user_1"
            every { authManager.getCurrentUserId() } returns userId
            coEvery { authManager.refreshAuthToken() } returns false

            val result = createWorker().doWork()

            assertEquals(Result.retry(), result)
        }

    @Test
    fun `doWork should return retry when IOException occurs`() =
        runBlocking {
            val userId = "user_1"
            every { authManager.getCurrentUserId() } returns userId
            // ФИКС: Токен валидный, но дальше падает сеть
            coEvery { authManager.refreshAuthToken() } returns true
            stubIncrementalSyncReady()
            coEvery { syncManager.syncIncremental(userId) } throws IOException("No network")

            val result = createWorker().doWork()

            assertEquals(Result.retry(), result)
        }

    @Test
    fun `doWork should return retry when FirebaseException occurs`() =
        runBlocking {
            val userId = "user_1"
            every { authManager.getCurrentUserId() } returns userId
            coEvery { authManager.refreshAuthToken() } returns true

            // Используем реальный инстанс FirebaseNetworkException вместо mockk
            val firebaseException = com.google.firebase.FirebaseNetworkException("StorageException: 403 Forbidden")
            stubIncrementalSyncReady()
            coEvery { syncManager.syncIncremental(userId) } throws firebaseException

            val result = createWorker().doWork()

            assertEquals(Result.retry(), result)
        }

    @Test
    fun `doWork should return failure on generic exception`() =
        runBlocking {
            val userId = "user_1"
            every { authManager.getCurrentUserId() } returns userId
            // ФИКС: Токен валидный, но дальше летит критическая ошибка
            coEvery { authManager.refreshAuthToken() } returns true
            stubIncrementalSyncReady()
            coEvery { syncManager.syncIncremental(userId) } throws RuntimeException("Fatal")

            val result = createWorker().doWork()

            assertEquals(Result.failure(), result)
        }
}

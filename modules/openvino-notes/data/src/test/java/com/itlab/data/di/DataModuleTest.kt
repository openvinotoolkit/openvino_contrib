package com.itlab.data.di

import android.content.Context
import androidx.work.ListenableWorker
import androidx.work.WorkManager
import androidx.work.WorkerParameters
import com.google.firebase.auth.FirebaseAuth
import com.google.firebase.storage.FirebaseStorage
import io.mockk.mockk
import org.junit.Test
import org.koin.android.ext.koin.androidContext
import org.koin.core.parameter.parametersOf
import org.koin.core.qualifier.named
import org.koin.dsl.koinApplication
import org.koin.dsl.module
import org.koin.test.KoinTest

class DataModuleTest : KoinTest {
    @Test
    fun `verify dataModule dependencies`() {
        val externalMocksModule =
            module {
                single<FirebaseAuth> { mockk(relaxed = true) }
                single<FirebaseStorage> { mockk(relaxed = true) }
                single<WorkManager> { mockk(relaxed = true) }
            }

        val app =
            koinApplication {
                allowOverride(true)
                androidContext(mockk<Context>(relaxed = true))
                modules(listOf(dataModule, externalMocksModule))
            }

        val koin = app.koin

        koin.get<com.itlab.domain.repository.NotesRepository>()
        koin.get<com.itlab.domain.repository.NoteFolderRepository>()
        koin.get<com.itlab.domain.repository.AuthRepository>()
        koin.get<com.itlab.domain.cloud.SyncScheduler>()
        koin.get<com.itlab.domain.cloud.SyncManager>()

        val workerClassName = com.itlab.data.cloud.SyncWorker::class.java.name

        val workerFactory =
            koin.getOrNull<ListenableWorker>(
                qualifier = named(workerClassName),
            ) {
                parametersOf(
                    mockk<Context>(relaxed = true),
                    mockk<WorkerParameters>(relaxed = true),
                )
            }

        assert(workerFactory != null)
    }
}

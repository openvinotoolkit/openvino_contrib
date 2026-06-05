package com.itlab.data.di

import androidx.room.Room
import com.itlab.data.BuildConfig
import com.itlab.data.cloud.AuthManager
import com.itlab.data.cloud.FirebaseCloudDataSource
import com.itlab.data.cloud.SharedPreferencesSyncCheckpointStore
import com.itlab.data.cloud.SyncCleaner
import com.itlab.data.cloud.SyncDaoContainer
import com.itlab.data.cloud.SyncManagerImpl
import com.itlab.data.cloud.SyncMappers
import com.itlab.data.cloud.SyncPuller
import com.itlab.data.cloud.SyncPusher
import com.itlab.data.cloud.SyncWorker
import com.itlab.data.cloud.WorkManagerSyncScheduler
import com.itlab.data.db.AppDatabase
import com.itlab.data.mapper.FolderEntityJsonConverter
import com.itlab.data.mapper.NoteEntityJsonConverter
import com.itlab.data.mapper.NoteFolderMapper
import com.itlab.data.mapper.NoteMapper
import com.itlab.data.repository.AuthRepositoryImpl
import com.itlab.data.repository.NoteFolderRepositoryImpl
import com.itlab.data.repository.NotesRepositoryImpl
import com.itlab.domain.cloud.CloudDataSource
import com.itlab.domain.cloud.SyncCheckpointStore
import com.itlab.domain.cloud.SyncManager
import com.itlab.domain.cloud.SyncScheduler
import com.itlab.domain.repository.AuthRepository
import com.itlab.domain.repository.NoteFolderRepository
import com.itlab.domain.repository.NotesRepository
import org.koin.androidx.workmanager.dsl.worker
import org.koin.dsl.module

val dataModule =
    module {
        single {
            com.google.firebase.auth.FirebaseAuth
                .getInstance()
        }

        single {
            com.google.firebase.storage.FirebaseStorage
                .getInstance()
        }

        single { androidx.work.WorkManager.getInstance(get()) }

        single { AuthManager(get()) }

        single {
            Room
                .databaseBuilder(
                    get(),
                    AppDatabase::class.java,
                    "notes_database",
                ).apply {
                    if (BuildConfig.DEBUG) {
                        fallbackToDestructiveMigration(true)
                    }
                }.build()
        }

        single { get<AppDatabase>().noteDao() }
        single { get<AppDatabase>().mediaDao() }
        single { get<AppDatabase>().folderDao() }

        single { NoteMapper() }
        single { NoteFolderMapper() }
        single { NoteEntityJsonConverter() }
        single { FolderEntityJsonConverter() }

        single<CloudDataSource> { FirebaseCloudDataSource(get()) }
        single { SyncDaoContainer(get(), get(), get()) }
        single { SyncMappers(get(), get(), get()) }
        single { SyncCleaner(get(), get(), get()) }
        single { SyncPuller(get(), get(), get(), get()) }
        single { SyncPusher(get(), get(), get()) }

        single<SyncManager> {
            SyncManagerImpl(
                get(),
                get(),
                get(),
            )
        }
        single<SyncScheduler> { WorkManagerSyncScheduler(get()) }
        single<SyncCheckpointStore> { SharedPreferencesSyncCheckpointStore(get()) }

        single<NotesRepository> { NotesRepositoryImpl(get(), get(), get()) }
        single<NoteFolderRepository> { NoteFolderRepositoryImpl(get(), get(), get(), get()) }
        single<AuthRepository> { AuthRepositoryImpl(get()) }

        worker { SyncWorker(get(), get(), get(), get(), get()) }
    }

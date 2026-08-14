// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package com.itlab.app

import android.app.Application
import androidx.work.Configuration
import com.itlab.assistant.api.NoteAssistant
import com.itlab.identity.api.IdentityService
import com.itlab.notes.api.NotesService
import com.itlab.settings.api.SettingsService
import com.itlab.sync.api.SyncService
import com.itlab.view.identity.signin.IdentityViewModel
import com.itlab.view.assistant.AssistantViewModel
import com.itlab.view.notes.editor.NoteEditorViewModel
import com.itlab.view.notes.list.NotesListViewModel
import com.itlab.view.settings.SettingsViewModel
import com.itlab.view.sync.SyncStatusViewModel
import org.koin.android.ext.koin.androidContext
import org.koin.core.context.startKoin
import org.koin.core.module.dsl.viewModel
import org.koin.dsl.module

class OpenVinoNotesApplication : Application(), Configuration.Provider {
    internal val composition: AppComposition by lazy { AppComposition(this) }

    override val workManagerConfiguration: Configuration
        get() = Configuration.Builder().setWorkerFactory(composition.workerFactory).build()

    override fun onCreate() {
        super.onCreate()
        val appModule = module {
            single<IdentityService> { composition.identityService }
            single<NotesService> { composition.notesService }
            single<SettingsService> { composition.settingsService }
            single<SyncService> { composition.syncService }
            single<NoteAssistant> { composition.noteAssistant }
            viewModel { NotesListViewModel(get()) }
            viewModel { IdentityViewModel(get()) }
            viewModel { SyncStatusViewModel(get()) }
            viewModel { SettingsViewModel(get()) }
            viewModel { AssistantViewModel(get()) }
            viewModel { NoteEditorViewModel(get()) }
        }
        startKoin {
            androidContext(this@OpenVinoNotesApplication)
            modules(appModule)
        }
    }
}

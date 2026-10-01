// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package com.openvino.notes.app

import android.app.Application
import androidx.work.Configuration
import com.openvino.notes.assistant.api.NoteAssistant
import com.openvino.notes.identity.api.IdentityService
import com.openvino.notes.notes.api.NotesService
import com.openvino.notes.notes.api.FolderService
import com.openvino.notes.settings.api.SettingsService
import com.openvino.notes.sync.api.SyncService
import com.openvino.notes.view.identity.signin.IdentityViewModel
import com.openvino.notes.view.assistant.AssistantViewModel
import com.openvino.notes.view.notes.editor.NoteEditorViewModel
import com.openvino.notes.view.notes.folders.FoldersViewModel
import com.openvino.notes.view.notes.list.NotesListViewModel
import com.openvino.notes.view.settings.SettingsViewModel
import com.openvino.notes.view.sync.SyncStatusViewModel
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
            single<FolderService> { composition.folderService }
            single<SettingsService> { composition.settingsService }
            single<SyncService> { composition.syncService }
            single<NoteAssistant> { composition.noteAssistant }
            viewModel { NotesListViewModel(get()) }
            viewModel { FoldersViewModel(get()) }
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

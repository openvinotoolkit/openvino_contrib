package com.itlab.view

import com.itlab.view.identity.signin.IdentityViewModel
import com.itlab.view.notes.list.NotesListViewModel
import com.itlab.view.settings.SettingsViewModel
import com.itlab.view.sync.SyncStatusViewModel
import org.koin.core.module.dsl.viewModel
import org.koin.dsl.module

val viewModule = module {
    viewModel { NotesListViewModel(get()) }
    viewModel { IdentityViewModel(get()) }
    viewModel { SyncStatusViewModel(get()) }
    viewModel { SettingsViewModel(get()) }
}

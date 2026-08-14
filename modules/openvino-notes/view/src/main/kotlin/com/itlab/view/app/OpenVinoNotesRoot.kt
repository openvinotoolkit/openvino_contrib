// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package com.itlab.view.app

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.itlab.identity.api.AuthenticationState
import com.itlab.view.assistant.AssistantScreen
import com.itlab.view.assistant.AssistantViewModel
import com.itlab.view.identity.signin.IdentityViewModel
import com.itlab.view.identity.signin.SignedOutScreen
import com.itlab.view.notes.list.NotesListScreen
import com.itlab.view.notes.list.NotesListViewModel
import com.itlab.view.settings.SettingsScreen
import com.itlab.view.settings.SettingsViewModel
import com.itlab.view.sync.SyncStatusScreen
import com.itlab.view.sync.SyncStatusViewModel

@Composable
fun OpenVinoNotesRoot(
    notes: NotesListViewModel,
    identity: IdentityViewModel,
    sync: SyncStatusViewModel,
    settings: SettingsViewModel,
    assistant: AssistantViewModel,
) {
    val authentication by identity.state.collectAsStateWithLifecycle()
    val appSettings by settings.settings.collectAsStateWithLifecycle()
    MaterialTheme {
        when (authentication) {
            AuthenticationState.SignedOut -> SignedOutScreen(identity::signIn)
            is AuthenticationState.SignedIn -> {
                var destination by remember { mutableStateOf(Destination.NOTES) }
                Scaffold(
                    bottomBar = {
                        AppNavigation(destination) { destination = it }
                    },
                ) { padding ->
                    when (destination) {
                        Destination.NOTES -> NotesListScreen(notes, Modifier, padding)
                        Destination.SYNC -> SyncStatusScreen(sync, Modifier, padding)
                        Destination.SETTINGS -> SettingsScreen(
                            settings = settings,
                            identity = identity,
                            themeName = appSettings.themeMode.name,
                            modifier = Modifier,
                            padding = padding,
                        )
                        Destination.ASSISTANT -> AssistantScreen(assistant, Modifier, padding)
                    }
                }
            }
        }
    }
}

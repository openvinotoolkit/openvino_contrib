// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package com.openvino.notes.view.app

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.openvino.notes.identity.api.AuthenticationState
import com.openvino.notes.view.assistant.AssistantScreen
import com.openvino.notes.view.assistant.AssistantViewModel
import com.openvino.notes.view.identity.signin.IdentityViewModel
import com.openvino.notes.view.identity.signin.SignedOutScreen
import com.openvino.notes.view.notes.list.NotesListScreen
import com.openvino.notes.view.notes.list.NotesListViewModel
import com.openvino.notes.view.settings.SettingsScreen
import com.openvino.notes.view.settings.SettingsViewModel
import com.openvino.notes.view.sync.SyncStatusScreen
import com.openvino.notes.view.sync.SyncStatusViewModel

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

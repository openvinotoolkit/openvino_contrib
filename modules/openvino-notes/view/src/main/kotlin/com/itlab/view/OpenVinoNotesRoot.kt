// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package com.itlab.view

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.itlab.identity.api.AuthenticationState
import com.itlab.view.identity.signin.IdentityViewModel
import com.itlab.view.notes.list.NotesListViewModel
import com.itlab.view.settings.SettingsViewModel
import com.itlab.view.sync.SyncStatusViewModel
import org.koin.compose.viewmodel.koinViewModel

private enum class Destination { NOTES, SYNC, SETTINGS, ASSISTANT }

@Composable
fun OpenVinoNotesRoot(
    notes: NotesListViewModel = koinViewModel(),
    identity: IdentityViewModel = koinViewModel(),
    sync: SyncStatusViewModel = koinViewModel(),
    settings: SettingsViewModel = koinViewModel(),
) {
    val authentication by identity.state.collectAsStateWithLifecycle()
    val appSettings by settings.settings.collectAsStateWithLifecycle()
    MaterialTheme {
        when (authentication) {
            AuthenticationState.SignedOut -> SignedOutScreen(identity)
            is AuthenticationState.SignedIn -> SignedInShell(notes, identity, sync, settings, appSettings.themeMode.name)
        }
    }
}

@Composable
private fun SignedOutScreen(identity: IdentityViewModel) {
    Column(Modifier.fillMaxSize().padding(24.dp), verticalArrangement = Arrangement.Center) {
        Text("OpenVINO Notes", style = MaterialTheme.typography.headlineMedium)
        Text("Sign in is intentionally unavailable until Google Identity is configured.", Modifier.padding(vertical = 16.dp))
        Button(onClick = { identity.signIn { } }) { Text("Sign in") }
    }
}

@Composable
private fun SignedInShell(
    notes: NotesListViewModel,
    identity: IdentityViewModel,
    sync: SyncStatusViewModel,
    settings: SettingsViewModel,
    themeName: String,
) {
    var destination by remember { mutableStateOf(Destination.NOTES) }
    Scaffold(
        bottomBar = {
            NavigationBar {
                Destination.entries.forEach { item ->
                    NavigationBarItem(
                        selected = destination == item,
                        onClick = { destination = item },
                        icon = { Text(item.name.take(1)) },
                        label = { Text(item.name.lowercase().replaceFirstChar(Char::uppercase)) },
                    )
                }
            }
        },
    ) { padding ->
        when (destination) {
            Destination.NOTES -> NotesScreen(notes, Modifier.padding(padding))
            Destination.SYNC -> SyncScreen(sync, Modifier.padding(padding))
            Destination.SETTINGS -> SettingsScreen(settings, identity, themeName, Modifier.padding(padding))
            Destination.ASSISTANT -> FeatureShell("Assistant", "Text and image OpenVINO adapters are present but require runtime and model assets.", Modifier.padding(padding))
        }
    }
}

@Composable
private fun NotesScreen(viewModel: NotesListViewModel, modifier: Modifier = Modifier) {
    val state by viewModel.state.collectAsStateWithLifecycle()
    Column(modifier.fillMaxSize().padding(16.dp)) {
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
            Text("Notes", style = MaterialTheme.typography.headlineMedium)
            Button(onClick = viewModel::createWelcomeNote) { Text("New") }
        }
        if (state.empty) Text("No notes yet.", Modifier.padding(top = 24.dp))
        LazyColumn { items(state.notes, key = { it.id.value }) { note -> Column(Modifier.padding(vertical = 8.dp)) { Text(note.title, style = MaterialTheme.typography.titleMedium); Text(note.body) } } }
    }
}

@Composable
private fun SyncScreen(viewModel: SyncStatusViewModel, modifier: Modifier = Modifier) {
    val state by viewModel.state.collectAsStateWithLifecycle()
    FeatureShell("Sync", "State: ${state.phase}. ${state.diagnosticCode.orEmpty()}", modifier) {
        Button(onClick = viewModel::syncNow) { Text("Sync now") }
    }
}

@Composable
private fun SettingsScreen(settings: SettingsViewModel, identity: IdentityViewModel, themeName: String, modifier: Modifier = Modifier) {
    FeatureShell("Settings", "Theme: $themeName", modifier) {
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(onClick = settings::cycleTheme) { Text("Change theme") }
            Button(onClick = identity::signOut) { Text("Sign out") }
        }
    }
}

@Composable
private fun FeatureShell(title: String, message: String, modifier: Modifier = Modifier, actions: @Composable () -> Unit = {}) {
    Column(modifier.fillMaxSize().padding(24.dp), verticalArrangement = Arrangement.spacedBy(16.dp)) {
        Text(title, style = MaterialTheme.typography.headlineMedium)
        Text(message)
        actions()
    }
}

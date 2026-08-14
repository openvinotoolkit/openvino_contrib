// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package com.itlab.view.settings

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.itlab.view.identity.signin.IdentityViewModel

@Composable
fun SettingsScreen(
    settings: SettingsViewModel,
    identity: IdentityViewModel,
    themeName: String,
    modifier: Modifier = Modifier,
    padding: PaddingValues = PaddingValues(),
) {
    Column(
        modifier.fillMaxSize().padding(padding).padding(24.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        Text("Settings", style = MaterialTheme.typography.headlineMedium)
        Text("Theme: $themeName")
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(onClick = settings::cycleTheme) { Text("Change theme") }
            Button(onClick = identity::authorizeDrive) { Text("Authorize Drive") }
            Button(onClick = identity::signOut) { Text("Sign out") }
        }
    }
}

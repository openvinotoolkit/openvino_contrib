// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package com.itlab.view.notes.list

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.itlab.notes.api.ContentItem

@Composable
fun NotesListScreen(
    viewModel: NotesListViewModel,
    modifier: Modifier = Modifier,
    padding: PaddingValues = PaddingValues(),
) {
    val state by viewModel.state.collectAsStateWithLifecycle()
    Column(modifier.fillMaxSize().padding(padding).padding(16.dp)) {
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
            Text("Notes", style = MaterialTheme.typography.headlineMedium)
            Button(onClick = viewModel::createWelcomeNote) { Text("New") }
        }
        if (state.empty) Text("No notes yet.", Modifier.padding(top = 24.dp))
        LazyColumn {
            items(state.notes, key = { it.id.value }) { note ->
                Column(Modifier.padding(vertical = 8.dp)) {
                    Text(note.title, style = MaterialTheme.typography.titleMedium)
                    Text(note.contentItems.filterIsInstance<ContentItem.Text>().joinToString("\n") { it.text })
                }
            }
        }
    }
}

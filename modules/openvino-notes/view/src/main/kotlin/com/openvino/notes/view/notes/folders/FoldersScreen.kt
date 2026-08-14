// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package com.openvino.notes.view.notes.folders

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

@Composable
fun FoldersScreen(
    viewModel: FoldersViewModel,
    modifier: Modifier = Modifier,
    padding: PaddingValues = PaddingValues(),
) {
    val state by viewModel.state.collectAsStateWithLifecycle()
    Column(modifier.fillMaxSize().padding(padding).padding(16.dp)) {
        Text("Folders", style = MaterialTheme.typography.headlineMedium)
        if (state.folders.isEmpty()) Text("No folders yet.", Modifier.padding(top = 24.dp))
        LazyColumn {
            items(state.folders, key = { it.id.value }) { folder ->
                Row(Modifier.fillMaxWidth().padding(vertical = 8.dp)) {
                    Text(folder.name, Modifier.weight(1f), style = MaterialTheme.typography.titleMedium)
                    Button(onClick = { viewModel.onAction(FoldersAction.Delete(folder.id)) }) {
                        Text("Delete")
                    }
                }
            }
        }
    }
}

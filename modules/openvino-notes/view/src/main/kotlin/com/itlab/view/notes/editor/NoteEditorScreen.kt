// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package com.itlab.view.notes.editor

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp

@Composable
fun NoteEditorScreen(state: NoteEditorUiState, onAction: (NoteEditorUiAction) -> Unit) {
    Column(Modifier.fillMaxSize().padding(16.dp)) {
        OutlinedTextField(state.title, { onAction(NoteEditorUiAction.ChangeTitle(it)) }, label = { Text("Title") })
        OutlinedTextField(state.text, { onAction(NoteEditorUiAction.ChangeText(it)) }, label = { Text("Text") })
        state.error?.let { Text(it) }
        Button(onClick = { onAction(NoteEditorUiAction.Save) }, enabled = !state.saving) { Text("Save") }
    }
}

// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package com.itlab.view.notes.list

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.itlab.notes.api.Note
import com.itlab.notes.api.NoteDraft
import com.itlab.notes.api.NotesService
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch

data class NotesListState(val notes: List<Note> = emptyList(), val empty: Boolean = true)

class NotesListViewModel(private val notes: NotesService) : ViewModel() {
    val state: StateFlow<NotesListState> = notes.observeNotes()
        .map { values -> NotesListState(values, values.isEmpty()) }
        .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5_000), NotesListState())

    fun createWelcomeNote() {
        viewModelScope.launch { notes.create(NoteDraft("Welcome", "OpenVINO Notes architecture is ready.")) }
    }
}

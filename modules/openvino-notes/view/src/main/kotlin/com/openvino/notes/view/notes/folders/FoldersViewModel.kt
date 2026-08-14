// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package com.openvino.notes.view.notes.folders

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.openvino.notes.notes.api.Folder
import com.openvino.notes.notes.api.FolderDraft
import com.openvino.notes.notes.api.FolderId
import com.openvino.notes.notes.api.FolderService
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch

data class FoldersState(val folders: List<Folder> = emptyList())

sealed interface FoldersAction {
    data class Create(val name: String) : FoldersAction
    data class Rename(val id: FolderId, val name: String) : FoldersAction
    data class Delete(val id: FolderId) : FoldersAction
}

class FoldersViewModel(private val folders: FolderService) : ViewModel() {
    val state: StateFlow<FoldersState> = folders.observeFolders()
        .map(::FoldersState)
        .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5_000), FoldersState())

    fun onAction(action: FoldersAction) {
        viewModelScope.launch {
            when (action) {
                is FoldersAction.Create -> folders.createFolder(FolderDraft(action.name))
                is FoldersAction.Rename -> folders.renameFolder(action.id, action.name)
                is FoldersAction.Delete -> folders.deleteFolder(action.id)
            }
        }
    }
}

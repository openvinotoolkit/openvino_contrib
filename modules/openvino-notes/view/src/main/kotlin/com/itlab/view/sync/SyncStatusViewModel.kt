// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package com.itlab.view.sync

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.itlab.sync.api.SyncReason
import com.itlab.sync.api.SyncService
import com.itlab.sync.api.SyncState
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch

data class SyncStatusUiState(val phase: String = "IDLE", val message: String? = null)
sealed interface SyncStatusAction { data object SyncNow : SyncStatusAction }

class SyncStatusViewModel(private val sync: SyncService) : ViewModel() {
    val state: StateFlow<SyncState> = sync.state
    val uiState: StateFlow<SyncStatusUiState> = state.map {
        SyncStatusUiState(it.phase.name, it.diagnosticCode)
    }.stateIn(viewModelScope, SharingStarted.WhileSubscribed(5_000), SyncStatusUiState())
    fun onAction(action: SyncStatusAction) = when (action) {
        SyncStatusAction.SyncNow -> syncNow()
    }
    fun syncNow() { viewModelScope.launch { sync.sync(SyncReason.USER) } }
}

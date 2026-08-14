package com.itlab.view.sync

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.itlab.sync.api.SyncReason
import com.itlab.sync.api.SyncService
import com.itlab.sync.api.SyncState
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch

class SyncStatusViewModel(private val sync: SyncService) : ViewModel() {
    val state: StateFlow<SyncState> = sync.state
    fun syncNow() { viewModelScope.launch { sync.sync(SyncReason.USER) } }
}

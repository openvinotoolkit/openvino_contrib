// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package com.openvino.notes.view.settings

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.openvino.notes.settings.api.AppSettings
import com.openvino.notes.settings.api.SettingsService
import com.openvino.notes.settings.api.ThemeMode
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch

class SettingsViewModel(private val settingsService: SettingsService) : ViewModel() {
    val settings: StateFlow<AppSettings> = settingsService.settings.stateIn(viewModelScope, SharingStarted.WhileSubscribed(5_000), AppSettings())
    fun onAction(action: SettingsUiAction) = when (action) {
        SettingsUiAction.CycleTheme -> cycleTheme()
    }
    fun cycleTheme() {
        val next = when (settings.value.themeMode) {
            ThemeMode.SYSTEM -> ThemeMode.LIGHT
            ThemeMode.LIGHT -> ThemeMode.DARK
            ThemeMode.DARK -> ThemeMode.SYSTEM
        }
        viewModelScope.launch { settingsService.setThemeMode(next) }
    }
}

sealed interface SettingsUiAction { data object CycleTheme : SettingsUiAction }

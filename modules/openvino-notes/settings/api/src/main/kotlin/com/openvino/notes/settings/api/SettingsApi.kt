// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package com.openvino.notes.settings.api

import kotlinx.coroutines.flow.Flow

enum class ThemeMode { SYSTEM, LIGHT, DARK }

data class AppSettings(
    val themeMode: ThemeMode = ThemeMode.SYSTEM,
    val dynamicColor: Boolean = true,
)

interface SettingsService {
    val settings: Flow<AppSettings>
    suspend fun setThemeMode(mode: ThemeMode)
    suspend fun setDynamicColor(enabled: Boolean)
}

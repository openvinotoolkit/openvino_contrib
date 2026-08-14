package com.itlab.settings.api

import kotlinx.coroutines.flow.MutableStateFlow

class FakeSettingsService(initial: AppSettings = AppSettings()) : SettingsService {
    private val state = MutableStateFlow(initial)
    override val settings = state
    override suspend fun setThemeMode(mode: ThemeMode) { state.value = state.value.copy(themeMode = mode) }
    override suspend fun setDynamicColor(enabled: Boolean) { state.value = state.value.copy(dynamicColor = enabled) }
}

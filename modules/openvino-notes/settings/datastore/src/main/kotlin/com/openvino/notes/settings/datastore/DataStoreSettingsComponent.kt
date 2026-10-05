// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package com.openvino.notes.settings.datastore

import android.content.Context
import androidx.datastore.core.DataStore
import androidx.datastore.preferences.core.Preferences
import androidx.datastore.preferences.core.booleanPreferencesKey
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.datastore.preferences.preferencesDataStoreFile
import androidx.datastore.preferences.core.PreferenceDataStoreFactory
import com.openvino.notes.settings.api.AppSettings
import com.openvino.notes.settings.api.SettingsService
import com.openvino.notes.settings.api.ThemeMode
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.flow.map

internal class DataStoreSettingsService(private val dataStore: DataStore<Preferences>) : SettingsService {
    override val settings = dataStore.data
        .catch { emit(androidx.datastore.preferences.core.emptyPreferences()) }
        .map { preferences ->
            AppSettings(
                themeMode = preferences[THEME]?.let { runCatching { ThemeMode.valueOf(it) }.getOrNull() } ?: ThemeMode.SYSTEM,
                dynamicColor = preferences[DYNAMIC_COLOR] ?: true,
            )
        }

    override suspend fun setThemeMode(mode: ThemeMode) { dataStore.edit { it[THEME] = mode.name } }
    override suspend fun setDynamicColor(enabled: Boolean) { dataStore.edit { it[DYNAMIC_COLOR] = enabled } }

    private companion object {
        val THEME = stringPreferencesKey("theme_mode")
        val DYNAMIC_COLOR = booleanPreferencesKey("dynamic_color")
    }
}

data class DataStoreSettingsComponent(val service: SettingsService) {
    companion object {
        fun create(context: Context): DataStoreSettingsComponent {
            val store = PreferenceDataStoreFactory.create { context.applicationContext.preferencesDataStoreFile("settings.preferences_pb") }
            return DataStoreSettingsComponent(DataStoreSettingsService(store))
        }
    }
}

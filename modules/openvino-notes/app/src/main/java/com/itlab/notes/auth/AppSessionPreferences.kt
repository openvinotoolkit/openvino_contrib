package com.itlab.notes.auth

import android.content.Context
import androidx.datastore.core.DataStore
import androidx.datastore.preferences.core.Preferences
import androidx.datastore.preferences.core.booleanPreferencesKey
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.preferencesDataStore
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map

internal val Context.appSessionDataStore: DataStore<Preferences> by preferencesDataStore(
    name = "app_session_preferences",
)

class AppSessionPreferences(
    private val context: Context,
) {
    val continueOffline: Flow<Boolean> =
        context.appSessionDataStore.data.map { prefs ->
            prefs[CONTINUE_OFFLINE] ?: false
        }

    suspend fun setContinueOffline(enabled: Boolean) {
        context.appSessionDataStore.edit { prefs ->
            prefs[CONTINUE_OFFLINE] = enabled
        }
    }

    private companion object {
        val CONTINUE_OFFLINE = booleanPreferencesKey("continue_offline")
    }
}

package com.itlab.notes.onboarding

import android.content.Context
import androidx.datastore.core.DataStore
import androidx.datastore.preferences.core.Preferences
import androidx.datastore.preferences.core.booleanPreferencesKey
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.preferencesDataStore
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map

internal val Context.onboardingDataStore: DataStore<Preferences> by preferencesDataStore(
    name = "onboarding_preferences",
)

class OnboardingPreferences(
    private val context: Context,
) {
    val welcomeCompleted: Flow<Boolean> =
        context.onboardingDataStore.data.map { prefs ->
            prefs[WELCOME_COMPLETED] ?: false
        }

    val tourCompleted: Flow<Boolean> =
        context.onboardingDataStore.data.map { prefs ->
            prefs[TOUR_COMPLETED] ?: false
        }

    suspend fun setWelcomeCompleted() {
        context.onboardingDataStore.edit { prefs ->
            prefs[WELCOME_COMPLETED] = true
        }
    }

    suspend fun setTourCompleted() {
        context.onboardingDataStore.edit { prefs ->
            prefs[TOUR_COMPLETED] = true
        }
    }

    private companion object {
        val WELCOME_COMPLETED = booleanPreferencesKey("welcome_completed")
        val TOUR_COMPLETED = booleanPreferencesKey("tour_completed")
    }
}

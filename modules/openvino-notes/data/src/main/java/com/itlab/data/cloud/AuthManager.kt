package com.itlab.data.cloud

import android.content.Context
import android.content.Intent
import com.firebase.ui.auth.AuthUI
import com.google.firebase.auth.FirebaseAuth
import kotlinx.coroutines.tasks.await
import timber.log.Timber

class AuthManager(
    private val auth: FirebaseAuth,
) {
    fun getSignInIntent(): Intent =
        AuthUI
            .getInstance()
            .createSignInIntentBuilder()
            .setAvailableProviders(
                listOf(
                    AuthUI.IdpConfig.EmailBuilder().build(),
                    AuthUI.IdpConfig.GoogleBuilder().build(),
                ),
            ).setIsSmartLockEnabled(false)
            .build()

    @Suppress("TooGenericExceptionCaught")
    suspend fun refreshAuthToken(): Boolean {
        val user = auth.currentUser ?: return false
        return try {
            Timber.d("Refreshing Firebase Auth token...")
            user.getIdToken(true).await()
            Timber.d("Firebase Auth token refreshed successfully")
            true
        } catch (e: Exception) {
            Timber.e(e, "Failed to refresh Firebase Auth token")
            false
        }
    }

    fun getCurrentUserId(): String? = auth.currentUser?.uid

    suspend fun signOut(context: Context) {
        AuthUI.getInstance().signOut(context).await()
    }
}

// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package com.openvino.notes.identity.google

import android.app.Activity
import com.openvino.notes.identity.api.AccessTokenOutcome
import com.openvino.notes.identity.api.AccessTokenProvider
import com.openvino.notes.identity.api.AuthenticationState
import com.openvino.notes.identity.api.DriveAuthorizationState
import com.openvino.notes.identity.api.IdentityOutcome
import com.openvino.notes.identity.api.IdentityService
import com.openvino.notes.identity.api.UserSession
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow

sealed interface GoogleSignInResult {
    data class Authenticated(val session: UserSession) : GoogleSignInResult
    data object Cancelled : GoogleSignInResult
    data class NotConfigured(val reason: String) : GoogleSignInResult
    data class Failed(val code: String) : GoogleSignInResult
}

sealed interface GoogleDriveAuthorizationResult {
    data object Authorized : GoogleDriveAuthorizationResult
    data object Cancelled : GoogleDriveAuthorizationResult
    data class NotConfigured(val reason: String) : GoogleDriveAuthorizationResult
    data class Failed(val code: String) : GoogleDriveAuthorizationResult
}

/** Activity-scoped launcher boundary implemented by the app once Google Identity is configured. */
interface GoogleIdentityLauncher {
    suspend fun launchSignIn(): GoogleSignInResult
    suspend fun authorizeDrive(): GoogleDriveAuthorizationResult
}

private class UnconfiguredGoogleIdentityLauncher : GoogleIdentityLauncher {
    override suspend fun launchSignIn() = GoogleSignInResult.NotConfigured(
        "Google Identity client is not configured",
    )
    override suspend fun authorizeDrive() = GoogleDriveAuthorizationResult.NotConfigured(
        "Drive OAuth scope is not configured",
    )
}

internal class GoogleIdentityService : IdentityService, AccessTokenProvider {
    private val authentication = MutableStateFlow<AuthenticationState>(AuthenticationState.Initializing)
    private val driveAuthorization = MutableStateFlow(DriveAuthorizationState.NOT_AUTHORIZED)
    private var cachedAccessToken: String? = null
    override val authenticationState: StateFlow<AuthenticationState> = authentication
    override val driveAuthorizationState: StateFlow<DriveAuthorizationState> = driveAuthorization
    override fun currentAccountKey() = (authentication.value as? AuthenticationState.SignedIn)?.session?.accountKey

    fun completeInitialization(session: UserSession?) {
        authentication.value = session?.let(AuthenticationState::SignedIn) ?: AuthenticationState.SignedOut
    }

    fun accept(result: GoogleSignInResult): IdentityOutcome = when (result) {
        is GoogleSignInResult.Authenticated -> {
            authentication.value = AuthenticationState.SignedIn(result.session)
            IdentityOutcome.Completed
        }
        GoogleSignInResult.Cancelled -> IdentityOutcome.Cancelled
        is GoogleSignInResult.NotConfigured -> IdentityOutcome.NotConfigured(result.reason)
        is GoogleSignInResult.Failed -> IdentityOutcome.Failed(result.code)
    }

    fun accept(result: GoogleDriveAuthorizationResult): IdentityOutcome = when (result) {
        GoogleDriveAuthorizationResult.Authorized -> {
            driveAuthorization.value = DriveAuthorizationState.AUTHORIZED
            IdentityOutcome.Completed
        }
        GoogleDriveAuthorizationResult.Cancelled -> IdentityOutcome.Cancelled
        is GoogleDriveAuthorizationResult.NotConfigured -> IdentityOutcome.NotConfigured(result.reason)
        is GoogleDriveAuthorizationResult.Failed -> IdentityOutcome.Failed(result.code)
    }

    override suspend fun signOut(): IdentityOutcome {
        invalidateAccessToken()
        authentication.value = AuthenticationState.SignedOut
        driveAuthorization.value = DriveAuthorizationState.NOT_AUTHORIZED
        return IdentityOutcome.Completed
    }

    override suspend fun disconnect(): IdentityOutcome = signOut()

    override suspend fun accessToken(): AccessTokenOutcome = when {
        currentAccountKey() == null -> AccessTokenOutcome.SignedOut
        driveAuthorization.value != DriveAuthorizationState.AUTHORIZED -> AccessTokenOutcome.NotAuthorized
        else -> AccessTokenOutcome.Failed("google.token_provider_not_connected")
    }

    override suspend fun invalidateAccessToken() {
        cachedAccessToken = null
    }
}

class GoogleIdentityUiController internal constructor(
    private val service: GoogleIdentityService,
    private val launcher: GoogleIdentityLauncher,
) {
    suspend fun signIn(): IdentityOutcome = service.accept(launcher.launchSignIn())
    suspend fun authorizeDrive(): IdentityOutcome = service.accept(launcher.authorizeDrive())
}

class GoogleIdentityComponent private constructor(
    private val service: GoogleIdentityService,
) {
    val identityService: IdentityService = service
    val accessTokenProvider: AccessTokenProvider = service

    /** Creates Activity-owned Google API state without retaining the Activity in this component. */
    fun createActivityController(activity: Activity): GoogleIdentityUiController =
        GoogleIdentityUiController(service, createLauncher(activity))

    internal fun createActivityController(launcher: GoogleIdentityLauncher): GoogleIdentityUiController =
        GoogleIdentityUiController(service, launcher)

    companion object {
        fun create(): GoogleIdentityComponent {
            val service = GoogleIdentityService()
            // Session persistence is not connected yet; keep the startup transition explicit.
            service.completeInitialization(null)
            return GoogleIdentityComponent(service)
        }

        private fun createLauncher(activity: Activity): GoogleIdentityLauncher {
            // Credential Manager and AuthorizationClient construction belongs here once configured.
            activity.applicationContext
            return UnconfiguredGoogleIdentityLauncher()
        }
    }
}

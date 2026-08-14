package com.itlab.identity.google

import com.itlab.identity.api.AccessTokenOutcome
import com.itlab.identity.api.AccessTokenProvider
import com.itlab.identity.api.AccountId
import com.itlab.identity.api.AuthenticationState
import com.itlab.identity.api.DriveAuthorizationState
import com.itlab.identity.api.IdentityOutcome
import com.itlab.identity.api.IdentityService
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow

interface GoogleIdentityLauncher {
    suspend fun launchSignIn(): IdentityOutcome
    suspend fun authorizeDrive(): IdentityOutcome
}

private object UnconfiguredGoogleIdentityLauncher : GoogleIdentityLauncher {
    override suspend fun launchSignIn() = IdentityOutcome.NotConfigured("Google Identity client is not configured")
    override suspend fun authorizeDrive() = IdentityOutcome.NotConfigured("Drive OAuth scope is not configured")
}

internal class GoogleIdentityService(private val launcher: GoogleIdentityLauncher) : IdentityService, AccessTokenProvider {
    private val authentication = MutableStateFlow<AuthenticationState>(AuthenticationState.SignedOut)
    private val driveAuthorization = MutableStateFlow(DriveAuthorizationState.NOT_AUTHORIZED)
    override val authenticationState: StateFlow<AuthenticationState> = authentication
    override val driveAuthorizationState: StateFlow<DriveAuthorizationState> = driveAuthorization
    override fun currentAccountId(): AccountId? = (authentication.value as? AuthenticationState.SignedIn)?.session?.accountId
    override suspend fun launchSignIn(): IdentityOutcome = launcher.launchSignIn()
    override suspend fun authorizeDrive(): IdentityOutcome = launcher.authorizeDrive()
    override suspend fun signOut(): IdentityOutcome {
        authentication.value = AuthenticationState.SignedOut
        driveAuthorization.value = DriveAuthorizationState.NOT_AUTHORIZED
        return IdentityOutcome.Completed
    }
    override suspend fun disconnect(): IdentityOutcome = signOut()
    override suspend fun accessToken(): AccessTokenOutcome = when {
        currentAccountId() == null -> AccessTokenOutcome.SignedOut
        driveAuthorization.value != DriveAuthorizationState.AUTHORIZED -> AccessTokenOutcome.NotAuthorized
        else -> AccessTokenOutcome.Failed("google.token_provider_not_connected")
    }
}

data class GoogleIdentityComponent(
    val identityService: IdentityService,
    val accessTokenProvider: AccessTokenProvider,
) {
    companion object {
        fun create(launcher: GoogleIdentityLauncher = UnconfiguredGoogleIdentityLauncher): GoogleIdentityComponent {
            val service = GoogleIdentityService(launcher)
            return GoogleIdentityComponent(service, service)
        }
    }
}

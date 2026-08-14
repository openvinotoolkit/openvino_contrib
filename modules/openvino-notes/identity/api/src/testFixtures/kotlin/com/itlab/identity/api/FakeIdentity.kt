package com.itlab.identity.api

import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow

class FakeIdentityService(
    initialAuthentication: AuthenticationState = AuthenticationState.SignedOut,
) : IdentityService, AccessTokenProvider {
    private val mutableAuthentication = MutableStateFlow(initialAuthentication)
    private val mutableDriveAuthorization = MutableStateFlow(DriveAuthorizationState.NOT_AUTHORIZED)

    override val authenticationState: StateFlow<AuthenticationState> = mutableAuthentication
    override val driveAuthorizationState: StateFlow<DriveAuthorizationState> = mutableDriveAuthorization

    override fun currentAccountId(): AccountId? =
        (authenticationState.value as? AuthenticationState.SignedIn)?.session?.accountId

    override suspend fun launchSignIn(): IdentityOutcome = IdentityOutcome.NotConfigured("fake requires an explicit session")
    override suspend fun authorizeDrive(): IdentityOutcome = IdentityOutcome.NotConfigured("fake requires explicit authorization")
    override suspend fun signOut(): IdentityOutcome {
        mutableAuthentication.value = AuthenticationState.SignedOut
        mutableDriveAuthorization.value = DriveAuthorizationState.NOT_AUTHORIZED
        return IdentityOutcome.Completed
    }
    override suspend fun disconnect(): IdentityOutcome = signOut()
    override suspend fun accessToken(): AccessTokenOutcome = when {
        currentAccountId() == null -> AccessTokenOutcome.SignedOut
        driveAuthorizationState.value != DriveAuthorizationState.AUTHORIZED -> AccessTokenOutcome.NotAuthorized
        else -> AccessTokenOutcome.Available("fake-token")
    }

    fun setSession(session: UserSession?) {
        mutableAuthentication.value = session?.let(AuthenticationState::SignedIn) ?: AuthenticationState.SignedOut
    }

    fun setDriveAuthorized(authorized: Boolean) {
        mutableDriveAuthorization.value = if (authorized) DriveAuthorizationState.AUTHORIZED else DriveAuthorizationState.NOT_AUTHORIZED
    }
}

// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package com.openvino.notes.identity.api

import com.openvino.notes.kernel.AccountKey
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow

class FakeIdentityService(
    initialAuthentication: AuthenticationState = AuthenticationState.SignedOut,
) : IdentityService, AccessTokenProvider {
    private val mutableAuthentication = MutableStateFlow(initialAuthentication)
    private val mutableDriveAuthorization = MutableStateFlow(DriveAuthorizationState.NOT_AUTHORIZED)

    override val authenticationState: StateFlow<AuthenticationState> = mutableAuthentication
    override val driveAuthorizationState: StateFlow<DriveAuthorizationState> = mutableDriveAuthorization
    var invalidatedAccessTokens: Int = 0
        private set

    override fun currentAccountKey(): AccountKey? =
        (authenticationState.value as? AuthenticationState.SignedIn)?.session?.accountKey

    override suspend fun signOut(): IdentityOutcome {
        mutableAuthentication.value = AuthenticationState.SignedOut
        mutableDriveAuthorization.value = DriveAuthorizationState.NOT_AUTHORIZED
        return IdentityOutcome.Completed
    }
    override suspend fun disconnect(): IdentityOutcome = signOut()
    override suspend fun accessToken(): AccessTokenOutcome = when {
        currentAccountKey() == null -> AccessTokenOutcome.SignedOut
        driveAuthorizationState.value != DriveAuthorizationState.AUTHORIZED -> AccessTokenOutcome.NotAuthorized
        else -> AccessTokenOutcome.Available("fake-token")
    }
    override suspend fun invalidateAccessToken() {
        invalidatedAccessTokens += 1
    }

    fun setSession(session: UserSession?) {
        mutableAuthentication.value = session?.let(AuthenticationState::SignedIn) ?: AuthenticationState.SignedOut
    }

    fun setDriveAuthorized(authorized: Boolean) {
        mutableDriveAuthorization.value = if (authorized) DriveAuthorizationState.AUTHORIZED else DriveAuthorizationState.NOT_AUTHORIZED
    }
}

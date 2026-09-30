// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package com.openvino.notes.identity.google

import com.openvino.notes.identity.api.AuthenticationState
import com.openvino.notes.identity.api.IdentityOutcome
import com.openvino.notes.identity.api.UserSession
import com.openvino.notes.identity.api.port.AccessTokenOutcome
import com.openvino.notes.kernel.AccountKey
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Test

class GoogleIdentityUiControllerTest {
    @Test fun `sign in cancellation remains distinct from failure`() = runTest {
        val component = GoogleIdentityComponent.create()

        assertEquals(
            IdentityOutcome.Cancelled,
            component.createActivityController(FakeLauncher(signInResult = GoogleSignInResult.Cancelled)).signIn(),
        )
        assertEquals(AuthenticationState.SignedOut, component.identityService.authenticationState.value)
    }

    @Test fun `authenticated result updates session through controller boundary`() = runTest {
        val session = UserSession(AccountKey("account"), "User", "user@example.test")
        val component = GoogleIdentityComponent.create()

        assertEquals(
            IdentityOutcome.Completed,
            component.createActivityController(
                FakeLauncher(signInResult = GoogleSignInResult.Authenticated(session)),
            ).signIn(),
        )
        assertEquals(AuthenticationState.SignedIn(session), component.identityService.authenticationState.value)
    }

    @Test fun `service exposes initialization before session restoration`() {
        val service = GoogleIdentityService()

        assertEquals(AuthenticationState.Initializing, service.authenticationState.value)
        service.completeInitialization(null)
        assertEquals(AuthenticationState.SignedOut, service.authenticationState.value)
    }

    @Test fun `token request for different account is rejected without consulting session`() = runTest {
        val sessionB = UserSession(AccountKey("account-b"), "User B", "b@example.test")
        val service = GoogleIdentityService()
        service.completeInitialization(sessionB)
        service.accept(GoogleDriveAuthorizationResult.Authorized)

        val outcome = service.accessToken(AccountKey("account-a"))

        assertEquals(AccessTokenOutcome.Failed("identity.account_mismatch"), outcome)
    }

    @Test fun `invalidating unknown account does not break current session`() = runTest {
        val service = GoogleIdentityService()
        val accountB = AccountKey("account-b")

        service.completeInitialization(UserSession(accountB, "B", "b@test"))
        service.accept(GoogleDriveAuthorizationResult.Authorized)

        service.invalidateAccessToken(AccountKey("account-a"))

        val outcome = service.accessToken(accountB)
        assertEquals(AccessTokenOutcome.Failed("google.token_provider_not_connected"), outcome)
    }

    @Test fun `available token does not leak through toString`() {
        val outcome = AccessTokenOutcome.Available("super-secret-token-12345")

        assertEquals(false, outcome.toString().contains("super-secret-token-12345"))
        assertEquals("Available(value=***)", outcome.toString())
    }

}

private class FakeLauncher(
    private val signInResult: GoogleSignInResult = GoogleSignInResult.NotConfigured("fake"),
    private val driveResult: GoogleDriveAuthorizationResult =
        GoogleDriveAuthorizationResult.NotConfigured("fake"),
) : GoogleIdentityLauncher {
    override suspend fun launchSignIn(): GoogleSignInResult = signInResult
    override suspend fun authorizeDrive(): GoogleDriveAuthorizationResult = driveResult
}

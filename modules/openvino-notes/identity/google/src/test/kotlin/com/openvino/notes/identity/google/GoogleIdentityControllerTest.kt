// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package com.openvino.notes.identity.google

import com.openvino.notes.identity.api.AuthenticationState
import com.openvino.notes.identity.api.IdentityOutcome
import com.openvino.notes.identity.api.UserSession
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
}

private class FakeLauncher(
    private val signInResult: GoogleSignInResult = GoogleSignInResult.NotConfigured("fake"),
    private val driveResult: GoogleDriveAuthorizationResult =
        GoogleDriveAuthorizationResult.NotConfigured("fake"),
) : GoogleIdentityLauncher {
    override suspend fun launchSignIn(): GoogleSignInResult = signInResult
    override suspend fun authorizeDrive(): GoogleDriveAuthorizationResult = driveResult
}

// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package com.itlab.identity.google

import com.itlab.identity.api.AuthenticationState
import com.itlab.identity.api.IdentityOutcome
import com.itlab.identity.api.UserSession
import com.itlab.kernel.AccountKey
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Test

class GoogleIdentityUiControllerTest {
    @Test fun `sign in cancellation remains distinct from failure`() = runTest {
        val component = GoogleIdentityComponent.create(
            FakeLauncher(signInResult = GoogleSignInResult.Cancelled),
        )

        assertEquals(IdentityOutcome.Cancelled, component.createActivityController().signIn())
        assertEquals(AuthenticationState.SignedOut, component.identityService.authenticationState.value)
    }

    @Test fun `authenticated result updates session through controller boundary`() = runTest {
        val session = UserSession(AccountKey("account"), "User", "user@example.test")
        val component = GoogleIdentityComponent.create(
            FakeLauncher(signInResult = GoogleSignInResult.Authenticated(session)),
        )

        assertEquals(IdentityOutcome.Completed, component.createActivityController().signIn())
        assertEquals(AuthenticationState.SignedIn(session), component.identityService.authenticationState.value)
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

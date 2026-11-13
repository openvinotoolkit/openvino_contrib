// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package com.openvino.notes.identity.google

import com.openvino.notes.identity.api.AuthenticationState
import com.openvino.notes.identity.api.DriveAuthorizationState
import com.openvino.notes.identity.api.IdentityOutcome
import com.openvino.notes.identity.api.UserSession
import com.openvino.notes.identity.api.port.AccessTokenOutcome
import com.openvino.notes.kernel.AccountKey
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.launch
import kotlinx.coroutines.test.runCurrent
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Test

@OptIn(ExperimentalCoroutinesApi::class)
class GoogleIdentityUiControllerTest {
    private val sessionA = UserSession(AccountKey("account-a"), "User A", "a@example.test")
    private val sessionB = UserSession(AccountKey("account-b"), "User B", "b@example.test")

    @Test fun `sign in cancellation remains distinct from failure`() =
        runTest {
            val component = GoogleIdentityComponent.create()

            assertEquals(
                IdentityOutcome.Cancelled,
                component.createActivityController(FakeLauncher(signInResult = GoogleSignInResult.Cancelled)).signIn(),
            )
            assertEquals(AuthenticationState.SignedOut, component.identityService.authenticationState.value)
        }

    @Test fun `authenticated result updates session through controller boundary`() =
        runTest {
            val component = GoogleIdentityComponent.create()

            assertEquals(
                IdentityOutcome.Completed,
                component
                    .createActivityController(
                        FakeLauncher(signInResult = GoogleSignInResult.Authenticated(sessionA)),
                    ).signIn(),
            )
            assertEquals(AuthenticationState.SignedIn(sessionA), component.identityService.authenticationState.value)
        }

    @Test fun `service exposes initialization before session restoration`() {
        val service = GoogleIdentityService()

        assertEquals(AuthenticationState.Initializing, service.authenticationState.value)
        service.completeInitialization(null)
        assertEquals(AuthenticationState.SignedOut, service.authenticationState.value)
    }

    @Test fun `token request for different account is rejected without consulting session`() =
        runTest {
            val service = GoogleIdentityService()
            service.completeInitialization(sessionB)

            service.accept(GoogleDriveAuthorizationResult.Authorized, service.captureContext())

            val outcome = service.accessToken(AccountKey("account-a"))

            assertEquals(AccessTokenOutcome.Failed("identity.account_mismatch"), outcome)
        }

    @Test fun `invalidating unknown account does not break current session`() =
        runTest {
            val service = GoogleIdentityService()
            val accountB = AccountKey("account-b")

            service.completeInitialization(UserSession(accountB, "B", "b@test"))
            service.accept(GoogleDriveAuthorizationResult.Authorized, service.captureContext())

            service.invalidateAccessToken(AccountKey("account-a"))

            val outcome = service.accessToken(accountB)
            assertEquals(AccessTokenOutcome.Failed("google.token_provider_not_connected"), outcome)
        }

    @Test fun `available token does not leak through toString`() {
        val outcome = AccessTokenOutcome.Available("super-secret-token-12345")

        assertEquals(false, outcome.toString().contains("super-secret-token-12345"))
        assertEquals("Available(value=***)", outcome.toString())
    }

    @Test
    fun `late authorization for A does not authorize B`() =
        runTest {
            val component = GoogleIdentityComponent.create()
            val launcher = ConcurrentFakeLauncher()
            val controller = component.createActivityController(launcher)

            val signInJobA = launch { controller.signIn() }
            launcher.signInDeferred.complete(GoogleSignInResult.Authenticated(sessionA))
            signInJobA.join()

            val authJob = launch { controller.authorizeDrive() }

            runCurrent()

            val signInJobB = launch { controller.signIn() }
            launcher.signInDeferred = CompletableDeferred()
            launcher.signInDeferred.complete(GoogleSignInResult.Authenticated(sessionB))
            signInJobB.join()

            launcher.driveAuthDeferred.complete(GoogleDriveAuthorizationResult.Authorized)
            authJob.join()

            assertEquals(DriveAuthorizationState.NOT_AUTHORIZED, component.identityService.driveAuthorizationState.value)
        }

    @Test
    fun `logout is preserved after late successful sign in`() =
        runTest {
            val component = GoogleIdentityComponent.create()
            val launcher = ConcurrentFakeLauncher()
            val controller = component.createActivityController(launcher)

            val signInJob = launch { controller.signIn() }
            runCurrent()

            component.identityService.signOut()

            launcher.signInDeferred.complete(GoogleSignInResult.Authenticated(sessionA))
            signInJob.join()

            assertEquals(AuthenticationState.SignedOut, component.identityService.authenticationState.value)
        }

    @Test
    fun `cancellation does not return Completed`() =
        runTest {
            val component = GoogleIdentityComponent.create()
            val launcher = ConcurrentFakeLauncher()
            val controller = component.createActivityController(launcher)

            val signInJob = launch { controller.signIn() }
            runCurrent()

            signInJob.cancel()

            assertEquals(AuthenticationState.SignedOut, component.identityService.authenticationState.value)
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

private class ConcurrentFakeLauncher : GoogleIdentityLauncher {
    var signInDeferred = CompletableDeferred<GoogleSignInResult>()
    val driveAuthDeferred = CompletableDeferred<GoogleDriveAuthorizationResult>()

    override suspend fun launchSignIn(): GoogleSignInResult = signInDeferred.await()

    override suspend fun authorizeDrive(): GoogleDriveAuthorizationResult = driveAuthDeferred.await()
}

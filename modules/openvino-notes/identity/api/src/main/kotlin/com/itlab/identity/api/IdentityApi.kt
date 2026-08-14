// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package com.itlab.identity.api

import kotlinx.coroutines.flow.StateFlow

@JvmInline
value class AccountId(val value: String)

data class UserSession(
    val accountId: AccountId,
    val displayName: String,
    val email: String,
)

sealed interface AuthenticationState {
    data object SignedOut : AuthenticationState
    data class SignedIn(val session: UserSession) : AuthenticationState
}

enum class DriveAuthorizationState { NOT_AUTHORIZED, AUTHORIZED }

sealed interface IdentityOutcome {
    data object Completed : IdentityOutcome
    data class NotConfigured(val reason: String) : IdentityOutcome
    data class Failed(val code: String) : IdentityOutcome
}

sealed interface AccessTokenOutcome {
    data class Available(val value: String) : AccessTokenOutcome
    data object SignedOut : AccessTokenOutcome
    data object NotAuthorized : AccessTokenOutcome
    data class Failed(val code: String) : AccessTokenOutcome
}

interface SessionReader {
    val authenticationState: StateFlow<AuthenticationState>
    fun currentAccountId(): AccountId?
}

interface AccessTokenProvider {
    suspend fun accessToken(): AccessTokenOutcome
}

interface IdentityService : SessionReader {
    val driveAuthorizationState: StateFlow<DriveAuthorizationState>
    suspend fun launchSignIn(): IdentityOutcome
    suspend fun authorizeDrive(): IdentityOutcome
    suspend fun signOut(): IdentityOutcome
    suspend fun disconnect(): IdentityOutcome
}

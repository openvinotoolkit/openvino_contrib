// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package com.openvino.notes.identity.api

import com.openvino.notes.kernel.AccountKey
import kotlinx.coroutines.flow.StateFlow

data class UserSession(
    val accountKey: AccountKey,
    val displayName: String,
    val email: String,
)

sealed interface AuthenticationState {
    data object Initializing : AuthenticationState
    data object SignedOut : AuthenticationState
    data class SignedIn(val session: UserSession) : AuthenticationState
}

enum class DriveAuthorizationState { NOT_AUTHORIZED, AUTHORIZED }

sealed interface IdentityOutcome {
    data object Completed : IdentityOutcome
    data object Cancelled : IdentityOutcome
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
    fun currentAccountKey(): AccountKey?
}

interface AccessTokenProvider {
    suspend fun accessToken(): AccessTokenOutcome
    suspend fun invalidateAccessToken()
}

interface IdentityService : SessionReader {
    val driveAuthorizationState: StateFlow<DriveAuthorizationState>
    suspend fun signOut(): IdentityOutcome
    suspend fun disconnect(): IdentityOutcome
}

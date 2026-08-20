// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package com.openvino.notes.identity.api

import com.openvino.notes.kernel.AccountKey
import com.openvino.notes.kernel.AccountScope
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.map

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

interface SessionReader : AccountScope {
    val authenticationState: StateFlow<AuthenticationState>
    override val activeAccountKey: Flow<AccountKey?>
        get() = authenticationState.map { state ->
            (state as? AuthenticationState.SignedIn)?.session?.accountKey
        }
    override fun currentAccountKey(): AccountKey?
}

interface IdentityService : SessionReader {
    val driveAuthorizationState: StateFlow<DriveAuthorizationState>
    suspend fun signOut(): IdentityOutcome
    suspend fun disconnect(): IdentityOutcome
}

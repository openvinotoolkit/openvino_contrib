// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package com.itlab.view.identity.signin

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.itlab.identity.api.AuthenticationState
import com.itlab.identity.api.IdentityOutcome
import com.itlab.identity.api.IdentityService
import com.itlab.view.app.AppUiEffect
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.flow.receiveAsFlow
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch

data class IdentityUiState(val signedIn: Boolean = false, val displayName: String? = null)

sealed interface IdentityUiAction {
    data object SignIn : IdentityUiAction
    data object AuthorizeDrive : IdentityUiAction
    data object SignOut : IdentityUiAction
}

class IdentityViewModel(private val identity: IdentityService) : ViewModel() {
    private val effectChannel = Channel<AppUiEffect>(Channel.BUFFERED)
    val effects: Flow<AppUiEffect> = effectChannel.receiveAsFlow()
    val state: StateFlow<AuthenticationState> = identity.authenticationState
    val uiState: StateFlow<IdentityUiState> = state.map { authentication ->
        when (authentication) {
            AuthenticationState.SignedOut -> IdentityUiState()
            is AuthenticationState.SignedIn -> IdentityUiState(true, authentication.session.displayName)
        }
    }.stateIn(viewModelScope, SharingStarted.WhileSubscribed(5_000), IdentityUiState())

    fun onAction(action: IdentityUiAction) = when (action) {
        IdentityUiAction.SignIn -> signIn()
        IdentityUiAction.AuthorizeDrive -> authorizeDrive()
        IdentityUiAction.SignOut -> signOut()
    }

    fun signIn() {
        effectChannel.trySend(AppUiEffect.LaunchGoogleSignIn)
    }

    fun authorizeDrive() {
        effectChannel.trySend(AppUiEffect.LaunchGoogleDriveAuthorization)
    }

    fun onActivityResult(outcome: IdentityOutcome) {
        val message = when (outcome) {
            IdentityOutcome.Completed -> return
            IdentityOutcome.Cancelled -> "Operation cancelled"
            is IdentityOutcome.NotConfigured -> outcome.reason
            is IdentityOutcome.Failed -> "Identity operation failed: ${outcome.code}"
        }
        effectChannel.trySend(AppUiEffect.Message(message))
    }

    fun signOut() {
        viewModelScope.launch { onActivityResult(identity.signOut()) }
    }
}

// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package com.itlab.view.identity.signin

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.itlab.identity.api.AuthenticationState
import com.itlab.identity.api.IdentityOutcome
import com.itlab.identity.api.IdentityService
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch

class IdentityViewModel(private val identity: IdentityService) : ViewModel() {
    val state: StateFlow<AuthenticationState> = identity.authenticationState
    fun signIn(onOutcome: (IdentityOutcome) -> Unit) {
        viewModelScope.launch { onOutcome(identity.launchSignIn()) }
    }
    fun signOut() { viewModelScope.launch { identity.signOut() } }
}

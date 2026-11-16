// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package com.openvino.notes.identity.api.port

import com.openvino.notes.kernel.AccountKey

sealed interface AccessTokenOutcome {
    data class Available(
        val value: String,
    ) : AccessTokenOutcome {
        override fun toString(): String = "Available(value=***)"
    }

    data object SignedOut : AccessTokenOutcome

    data object NotAuthorized : AccessTokenOutcome

    data class Failed(
        val code: String,
    ) : AccessTokenOutcome
}

/** Infrastructure credential boundary. Presentation code must use IdentityService instead. */
interface AccessTokenProvider {
    suspend fun accessToken(accountKey: AccountKey): AccessTokenOutcome

    suspend fun invalidateAccessToken(accountKey: AccountKey)
}

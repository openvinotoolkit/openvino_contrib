// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package com.openvino.notes.identity.api.port

import com.openvino.notes.identity.api.AccessTokenOutcome

/** Infrastructure credential boundary. Presentation code must use IdentityService instead. */
interface AccessTokenProvider {
    suspend fun accessToken(): AccessTokenOutcome
    suspend fun invalidateAccessToken()
}

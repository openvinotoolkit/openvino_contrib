// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package com.openvino.notes.view.identity.signin

import com.openvino.notes.identity.api.FakeIdentityService
import com.openvino.notes.view.app.AppUiEffect
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Test

class IdentityViewModelTest {
    @Test fun `sign in and authorization are separate one shot effects`() = runTest {
        val viewModel = IdentityViewModel(FakeIdentityService())

        viewModel.signIn()
        assertEquals(AppUiEffect.LaunchGoogleSignIn, viewModel.effects.first())

        viewModel.authorizeDrive()
        assertEquals(AppUiEffect.LaunchGoogleDriveAuthorization, viewModel.effects.first())
    }
}

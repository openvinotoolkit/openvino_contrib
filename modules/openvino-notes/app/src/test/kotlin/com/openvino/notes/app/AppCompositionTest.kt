// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package com.openvino.notes.app

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import org.junit.Assert.assertNotNull
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [35])
class AppCompositionTest {
    @Test fun `composition exposes public feature contracts`() {
        val context = ApplicationProvider.getApplicationContext<Context>()
        AppComposition(context).use { composition ->
            assertNotNull(composition.notesService)
            assertNotNull(composition.folderService)
            assertNotNull(composition.identityService)
            assertNotNull(composition.syncService)
            assertNotNull(composition.noteAssistant)
        }
    }
}

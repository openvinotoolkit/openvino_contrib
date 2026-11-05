// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package com.openvino.notes.app

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import org.junit.After
import org.junit.Assert.assertNotNull
import org.junit.Test
import org.junit.runner.RunWith
import org.koin.core.context.stopKoin
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [35])
class AppCompositionTest {
    @After fun tearDown() {
        stopKoin()
    }

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

    @Test fun `composition runs without network and google configuration`() {
        val context = ApplicationProvider.getApplicationContext<Context>()
        AppComposition(context).use { composition ->
            assertNotNull(composition.identityService)
            assertNotNull(composition.workerFactory)
            assertNotNull(composition.syncScheduler)
        }
    }
}

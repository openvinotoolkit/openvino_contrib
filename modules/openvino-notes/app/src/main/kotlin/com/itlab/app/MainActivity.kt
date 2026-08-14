// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package com.itlab.app

import android.os.Bundle
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.runtime.LaunchedEffect
import com.itlab.view.app.AppUiEffect
import com.itlab.view.app.OpenVinoNotesRoot
import com.itlab.view.assistant.AssistantViewModel
import com.itlab.view.identity.signin.IdentityViewModel
import com.itlab.view.notes.list.NotesListViewModel
import com.itlab.view.settings.SettingsViewModel
import com.itlab.view.sync.SyncStatusViewModel
import org.koin.compose.viewmodel.koinViewModel

class MainActivity : ComponentActivity() {
    private val identityController by lazy {
        (application as OpenVinoNotesApplication).composition.createActivityIdentityController()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            val notes = koinViewModel<NotesListViewModel>()
            val identity = koinViewModel<IdentityViewModel>()
            val sync = koinViewModel<SyncStatusViewModel>()
            val settings = koinViewModel<SettingsViewModel>()
            val assistant = koinViewModel<AssistantViewModel>()

            LaunchedEffect(identity) {
                identity.effects.collect { effect ->
                    when (effect) {
                        AppUiEffect.LaunchGoogleSignIn ->
                            identity.onActivityResult(identityController.signIn())
                        AppUiEffect.LaunchGoogleDriveAuthorization ->
                            identity.onActivityResult(identityController.authorizeDrive())
                        is AppUiEffect.Message -> Toast.makeText(
                            this@MainActivity,
                            effect.text,
                            Toast.LENGTH_SHORT,
                        ).show()
                    }
                }
            }

            OpenVinoNotesRoot(notes, identity, sync, settings, assistant)
        }
    }
}

// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package com.openvino.notes.app

import android.os.Bundle
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.runtime.LaunchedEffect
import com.openvino.notes.view.app.AppUiEffect
import com.openvino.notes.view.app.OpenVinoNotesRoot
import com.openvino.notes.view.assistant.AssistantViewModel
import com.openvino.notes.view.identity.signin.IdentityViewModel
import com.openvino.notes.view.notes.list.NotesListViewModel
import com.openvino.notes.view.notes.folders.FoldersViewModel
import com.openvino.notes.view.settings.SettingsViewModel
import com.openvino.notes.view.sync.SyncStatusViewModel
import org.koin.compose.viewmodel.koinViewModel

class MainActivity : ComponentActivity() {
    private val identityController by lazy {
        (application as OpenVinoNotesApplication).composition.createActivityIdentityController(this)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            val notes = koinViewModel<NotesListViewModel>()
            val folders = koinViewModel<FoldersViewModel>()
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

            OpenVinoNotesRoot(notes, folders, identity, sync, settings, assistant)
        }
    }
}

// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package com.openvino.notes.view.app

import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onAllNodesWithText
import androidx.compose.ui.test.onLast
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import com.openvino.notes.assistant.api.FakeNoteAssistant
import com.openvino.notes.identity.api.AuthenticationState
import com.openvino.notes.identity.api.FakeIdentityService
import com.openvino.notes.identity.api.UserSession
import com.openvino.notes.kernel.AccountKey
import com.openvino.notes.notes.api.FakeFolderService
import com.openvino.notes.notes.api.FakeNotesService
import com.openvino.notes.settings.api.FakeSettingsService
import com.openvino.notes.sync.api.FakeSyncService
import com.openvino.notes.view.assistant.AssistantViewModel
import com.openvino.notes.view.identity.signin.IdentityViewModel
import com.openvino.notes.view.notes.folders.FoldersViewModel
import com.openvino.notes.view.notes.list.NotesListViewModel
import com.openvino.notes.view.settings.SettingsViewModel
import com.openvino.notes.view.sync.SyncStatusViewModel
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [35])
class AppNavigationTest {
    @get:Rule
    val composeTestRule = createComposeRule()

    @Test
    fun `navigation switches tabs and user action invokes service once`() {
        val notesService = FakeNotesService()
        val folderService = FakeFolderService()
        val session = UserSession(AccountKey("user-1"), "Test User", "test@example.com")
        val identityService = FakeIdentityService(AuthenticationState.SignedIn(session))
        val syncService = FakeSyncService()
        val settingsService = FakeSettingsService()
        val assistantService = FakeNoteAssistant()

        val notesVm = NotesListViewModel(notesService)
        val foldersVm = FoldersViewModel(folderService)
        val identityVm = IdentityViewModel(identityService)
        val syncVm = SyncStatusViewModel(syncService)
        val settingsVm = SettingsViewModel(settingsService)
        val assistantVm = AssistantViewModel(assistantService)

        composeTestRule.setContent {
            OpenVinoNotesRoot(
                notes = notesVm,
                folders = foldersVm,
                identity = identityVm,
                sync = syncVm,
                settings = settingsVm,
                assistant = assistantVm,
            )
        }

        // Verify initial screen (Notes)
        composeTestRule.onNodeWithText("No notes yet.").assertIsDisplayed()

        // User action: Click "New" to create welcome note
        composeTestRule.onNodeWithText("New").performClick()

        // Check service was invoked once (note created in FakeNotesService)
        assertEquals(1, notesService.currentNotes.size)
        assertEquals("Welcome", notesService.currentNotes.first().title)

        // Navigation: Click "Folders" in bottom navigation
        composeTestRule.onAllNodesWithText("Folders").onLast().performClick()
        composeTestRule.onNodeWithText("No folders yet.").assertIsDisplayed()

        // Navigation: Click "Sync" in bottom navigation
        composeTestRule.onAllNodesWithText("Sync").onLast().performClick()
        composeTestRule.onNodeWithText("Sync now").assertIsDisplayed()

        // Navigation: Click "Settings" in bottom navigation
        composeTestRule.onAllNodesWithText("Settings").onLast().performClick()
        composeTestRule.onNodeWithText("Change theme").assertIsDisplayed()

        // Navigation: Click "Assistant" in bottom navigation
        composeTestRule.onAllNodesWithText("Assistant").onLast().performClick()
        composeTestRule.onNodeWithText("Select an AI operation").assertIsDisplayed()
    }

    @Test
    fun `signed out state renders sign in button and triggers sign in effect`() = runTest {
        val identityService = FakeIdentityService(AuthenticationState.SignedOut)
        val identityVm = IdentityViewModel(identityService)

        composeTestRule.setContent {
            OpenVinoNotesRoot(
                notes = NotesListViewModel(FakeNotesService()),
                folders = FoldersViewModel(FakeFolderService()),
                identity = identityVm,
                sync = SyncStatusViewModel(FakeSyncService()),
                settings = SettingsViewModel(FakeSettingsService()),
                assistant = AssistantViewModel(FakeNoteAssistant()),
            )
        }

        composeTestRule.onNodeWithText("Sign in").assertIsDisplayed()
        composeTestRule.onNodeWithText("Sign in").performClick()

        assertEquals(AppUiEffect.LaunchGoogleSignIn, identityVm.effects.first())
    }
}

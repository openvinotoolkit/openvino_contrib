// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package com.openvino.notes.view.notes.editor

import com.openvino.notes.kernel.AccountKey
import com.openvino.notes.notes.api.AttachmentId
import com.openvino.notes.notes.api.ContentItem
import com.openvino.notes.notes.api.ContentItemId
import com.openvino.notes.notes.api.FakeNotesService
import com.openvino.notes.notes.api.Note
import com.openvino.notes.notes.api.NoteId
import java.time.Instant
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.test.UnconfinedTestDispatcher
import kotlinx.coroutines.test.advanceUntilIdle
import kotlinx.coroutines.test.resetMain
import kotlinx.coroutines.test.runTest
import kotlinx.coroutines.test.setMain
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Before
import org.junit.Test

@OptIn(ExperimentalCoroutinesApi::class)
class NoteEditorViewModelTest {
    private val dispatcher = UnconfinedTestDispatcher()

    @Before fun setUp() = Dispatchers.setMain(dispatcher)
    @After fun tearDown() = Dispatchers.resetMain()

    @Test fun `text edit preserves unrelated multimodal content`() = runTest {
        val firstText = ContentItem.Text(ContentItemId("text-a"), "First")
        val image = ContentItem.Image(ContentItemId("image"), AttachmentId("image-content"))
        val file = ContentItem.File(ContentItemId("file"), AttachmentId("file-content"))
        val secondText = ContentItem.Text(ContentItemId("text-b"), "Second")
        val link = ContentItem.Link(ContentItemId("link"), "https://example.com")
        val note = Note(
            id = NoteId("note"),
            accountKey = AccountKey("account"),
            title = "Multimodal",
            contentItems = listOf(firstText, image, file, secondText, link),
            createdAt = Instant.EPOCH,
            updatedAt = Instant.EPOCH,
        )
        val notes = FakeNotesService(listOf(note))
        val viewModel = NoteEditorViewModel(notes)

        viewModel.onAction(NoteEditorUiAction.Load(note.id))
        advanceUntilIdle()
        viewModel.onAction(NoteEditorUiAction.ChangeText("Updated first"))
        viewModel.onAction(NoteEditorUiAction.Save)
        advanceUntilIdle()

        assertEquals(
            listOf(firstText.copy(text = "Updated first"), image, file, secondText, link),
            notes.get(note.id)?.contentItems,
        )
    }
}

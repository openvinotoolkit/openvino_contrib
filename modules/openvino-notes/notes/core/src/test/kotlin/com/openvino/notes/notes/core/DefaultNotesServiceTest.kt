// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package com.openvino.notes.notes.core

import com.openvino.notes.identity.api.FakeIdentityService
import com.openvino.notes.identity.api.UserSession
import com.openvino.notes.kernel.AccountKey
import com.openvino.notes.notes.api.ContentItem
import com.openvino.notes.notes.api.ContentItemId
import com.openvino.notes.notes.api.FakeNotesRepository
import com.openvino.notes.notes.api.FakeFolderRepository
import com.openvino.notes.notes.api.FieldUpdate
import com.openvino.notes.notes.api.NoteDraft
import com.openvino.notes.notes.api.NoteId
import com.openvino.notes.notes.api.NoteMutationOutcome
import com.openvino.notes.notes.api.UpdateNoteCommand
import java.time.Instant
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class DefaultNotesServiceTest {
    private val content = listOf(ContentItem.Text(ContentItemId("body"), "Body"))

    @Test fun `create requires a session`() = runTest {
        val service = DefaultNotesService(
            FakeIdentityService(),
            FakeNotesRepository(),
            FakeFolderRepository(),
            { Instant.EPOCH },
        )
        assertEquals(NoteMutationOutcome.SignedOut, service.create(NoteDraft("Title", content)))
    }

    @Test fun `patch preserves fields that are not supplied`() = runTest {
        val accountKey = AccountKey("account")
        val identity = FakeIdentityService().apply {
            setSession(UserSession(accountKey, "User", "u@example.test"))
        }
        val repository = FakeNotesRepository()
        var now = Instant.EPOCH
        val service = DefaultNotesService(identity, repository, FakeFolderRepository(), { now }) { NoteId("note-1") }
        service.create(NoteDraft("  Title  ", content, isFavorite = true))

        now = Instant.ofEpochSecond(1)
        val outcome = service.update(
            UpdateNoteCommand(NoteId("note-1"), summary = FieldUpdate.Set("Summary")),
        )

        assertTrue(outcome is NoteMutationOutcome.Saved)
        val saved = repository.find(accountKey, NoteId("note-1"))
        assertEquals("Title", saved?.title)
        assertEquals(content, saved?.contentItems)
        assertEquals(true, saved?.isFavorite)
        assertEquals("Summary", saved?.summary)
        assertEquals(Instant.EPOCH, saved?.createdAt)
        assertEquals(now, saved?.updatedAt)
    }
}

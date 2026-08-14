// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package com.openvino.notes.notes.core

import com.openvino.notes.identity.api.FakeIdentityService
import com.openvino.notes.identity.api.UserSession
import com.openvino.notes.kernel.AccountKey
import com.openvino.notes.notes.api.ContentItem
import com.openvino.notes.notes.api.ContentItemId
import com.openvino.notes.notes.api.FakeFolderRepository
import com.openvino.notes.notes.api.FakeNotesRepository
import com.openvino.notes.notes.api.FolderDraft
import com.openvino.notes.notes.api.FolderId
import com.openvino.notes.notes.api.FolderMutationOutcome
import com.openvino.notes.notes.api.MoveNoteOutcome
import com.openvino.notes.notes.api.NoteDraft
import com.openvino.notes.notes.api.NoteId
import java.time.Instant
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class DefaultFolderServiceTest {
    @Test fun `folder lifecycle validates moves and rejects deleting a non-empty folder`() = runTest {
        val accountKey = AccountKey("account")
        val identity = FakeIdentityService().apply {
            setSession(UserSession(accountKey, "User", "user@example.test"))
        }
        val notesRepository = FakeNotesRepository()
        val folderRepository = FakeFolderRepository()
        val notes = DefaultNotesService(
            identity,
            notesRepository,
            folderRepository,
            { Instant.EPOCH },
        ) { NoteId("note") }
        val folders = DefaultFolderService(
            identity,
            folderRepository,
            notesRepository,
            notes,
            { Instant.EPOCH },
        ) { FolderId("folder") }

        val created = folders.createFolder(FolderDraft(" Projects ")) as FolderMutationOutcome.Saved
        assertEquals("Projects", created.folder.name)
        assertEquals(listOf(created.folder), folders.observeFolders().first())

        notes.create(
            NoteDraft(
                title = "Note",
                contentItems = listOf(ContentItem.Text(ContentItemId("body"), "Body")),
                folderId = created.folder.id,
            ),
        )
        assertEquals(FolderMutationOutcome.NotEmpty(1), folders.deleteFolder(created.folder.id))

        assertTrue(folders.moveNote(NoteId("note"), null) is MoveNoteOutcome.Moved)
        assertEquals(FolderMutationOutcome.Deleted, folders.deleteFolder(created.folder.id))
    }

    @Test fun `move rejects an unknown folder without changing the note`() = runTest {
        val accountKey = AccountKey("account")
        val identity = FakeIdentityService().apply {
            setSession(UserSession(accountKey, "User", "user@example.test"))
        }
        val notesRepository = FakeNotesRepository()
        val folderRepository = FakeFolderRepository()
        val notes = DefaultNotesService(
            identity,
            notesRepository,
            folderRepository,
            { Instant.EPOCH },
        ) { NoteId("note") }
        val folders = DefaultFolderService(
            identity,
            folderRepository,
            notesRepository,
            notes,
            { Instant.EPOCH },
        )
        notes.create(NoteDraft("Note", listOf(ContentItem.Text(ContentItemId("body"), "Body"))))

        assertEquals(MoveNoteOutcome.FolderNotFound, folders.moveNote(NoteId("note"), FolderId("missing")))
        assertEquals(null, notes.get(NoteId("note"))?.folderId)
    }
}

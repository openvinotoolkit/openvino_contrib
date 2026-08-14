// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package com.openvino.notes.notes.room

import android.content.Context
import androidx.room.Room
import androidx.test.core.app.ApplicationProvider
import com.openvino.notes.kernel.AccountKey
import com.openvino.notes.notes.api.ContentItem
import com.openvino.notes.notes.api.ContentItemId
import com.openvino.notes.notes.api.Note
import com.openvino.notes.notes.api.NoteId
import com.openvino.notes.notes.api.NoteTag
import com.openvino.notes.notes.api.RemoteApplyResult
import com.openvino.notes.notes.api.RemoteNoteChange
import com.openvino.notes.notes.api.RemoteRevision
import java.time.Instant
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [35])
class RoomNotesRepositoryTest {
    @Test fun `save preserves the complete note and writes an outbox snapshot`() = runTest {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val database = Room.inMemoryDatabaseBuilder(context, NotesDatabase::class.java).allowMainThreadQueries().build()
        try {
            val repository = RoomNotesRepository(database.notesDao())
            val accountKey = AccountKey("account")
            val note = Note(
                id = NoteId("note"),
                accountKey = accountKey,
                title = "Title",
                contentItems = listOf(ContentItem.Text(ContentItemId("body"), "Body")),
                tags = setOf(NoteTag("important")),
                isFavorite = true,
                summary = "Summary",
                createdAt = Instant.EPOCH,
                updatedAt = Instant.EPOCH,
            )

            repository.save(note)

            assertEquals(note, repository.find(accountKey, note.id))
            assertEquals(listOf(note), repository.pendingChanges(accountKey).map { it.payload })
        } finally {
            database.close()
        }
    }

    @Test fun `remote changes preserve conflicts and tombstones without outbox loops`() = runTest {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val database = Room.inMemoryDatabaseBuilder(context, NotesDatabase::class.java).allowMainThreadQueries().build()
        try {
            val repository = RoomNotesRepository(database.notesDao())
            val accountKey = AccountKey("account")
            val note = Note(
                id = NoteId("remote-note"),
                accountKey = accountKey,
                title = "Remote",
                contentItems = listOf(ContentItem.Text(ContentItemId("body"), "Body")),
                createdAt = Instant.EPOCH,
                updatedAt = Instant.EPOCH,
            )
            val revision1 = RemoteRevision("1")
            val revision2 = RemoteRevision("2")

            assertEquals(
                listOf(RemoteApplyResult.Applied(note.id, revision1)),
                repository.applyRemote(accountKey, listOf(RemoteNoteChange.Upsert(note, null, revision1))),
            )
            assertEquals(emptyList<NoteId>(), repository.pendingChanges(accountKey).map { it.noteId })
            assertEquals(
                listOf(RemoteApplyResult.TombstoneApplied(note.id, revision2)),
                repository.applyRemote(accountKey, listOf(RemoteNoteChange.Tombstone(note.id, revision2))),
            )
            assertEquals(null, repository.find(accountKey, note.id))

            repository.save(note.copy(title = "Local"))
            assertEquals(
                listOf(RemoteApplyResult.Conflict(note.id, revision2, revision2)),
                repository.applyRemote(accountKey, listOf(RemoteNoteChange.Upsert(note, revision1, revision2))),
            )
        } finally {
            database.close()
        }
    }
}

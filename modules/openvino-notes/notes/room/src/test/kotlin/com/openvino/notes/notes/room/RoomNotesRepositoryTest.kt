// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package com.openvino.notes.notes.room

import android.content.Context
import androidx.room.Room
import androidx.test.core.app.ApplicationProvider
import com.openvino.notes.kernel.AccountKey
import com.openvino.notes.notes.api.AttachmentId
import com.openvino.notes.notes.api.AttachmentMetadata
import com.openvino.notes.notes.api.BinarySource
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
import org.junit.Assert.assertArrayEquals
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
            val repository = RoomNotesRepository(
                database.notesDao(),
                FileAttachmentContentStore(context.cacheDir.resolve("notes-room-save-test")),
            )
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
            val repository = RoomNotesRepository(
                database.notesDao(),
                FileAttachmentContentStore(context.cacheDir.resolve("notes-room-remote-test")),
            )
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

    @Test fun `attachment files are opened through the port and removed with metadata or note deletion`() = runTest {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val database = Room.inMemoryDatabaseBuilder(context, NotesDatabase::class.java).allowMainThreadQueries().build()
        val root = context.cacheDir.resolve("notes-room-attachment-test").apply { deleteRecursively() }
        val content = FileAttachmentContentStore(root)
        try {
            val repository = RoomNotesRepository(database.notesDao(), content)
            val accountKey = AccountKey("account")
            val noteId = NoteId("note")
            val itemId = ContentItemId("image")
            val firstId = AttachmentId("first")
            val secondId = AttachmentId("second")
            val first = AttachmentMetadata(firstId, noteId, itemId, "first.png", "image/png", 3)
            val second = AttachmentMetadata(secondId, noteId, itemId, "second.png", "image/png", 2)
            content.put(accountKey, first, BinarySource { byteArrayOf(1, 2, 3) })
            content.put(accountKey, second, BinarySource { byteArrayOf(4, 5) })
            val note = Note(
                id = noteId,
                accountKey = accountKey,
                title = "Images",
                contentItems = listOf(ContentItem.Image(itemId, secondId)),
                attachments = listOf(first, second),
                createdAt = Instant.EPOCH,
                updatedAt = Instant.EPOCH,
            )

            repository.save(note)
            assertArrayEquals(byteArrayOf(1, 2, 3), content.open(accountKey, firstId)?.read())

            repository.save(note.copy(attachments = listOf(second)))
            assertEquals(null, content.open(accountKey, firstId))
            repository.delete(accountKey, noteId)
            assertEquals(null, content.open(accountKey, secondId))
        } finally {
            database.close()
            root.deleteRecursively()
        }
    }
}

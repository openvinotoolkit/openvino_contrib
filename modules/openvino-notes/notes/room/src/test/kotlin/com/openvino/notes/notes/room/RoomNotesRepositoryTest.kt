// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package com.openvino.notes.notes.room

import android.content.Context
import androidx.room.Room
import androidx.test.core.app.ApplicationProvider
import com.openvino.notes.kernel.AccountKey
import com.openvino.notes.kernel.AppDispatchers
import com.openvino.notes.notes.api.AttachmentId
import com.openvino.notes.notes.api.AttachmentMetadata
import com.openvino.notes.notes.api.ContentItem
import com.openvino.notes.notes.api.ContentItemId
import com.openvino.notes.notes.api.Note
import com.openvino.notes.notes.api.NoteId
import com.openvino.notes.notes.api.NoteTag
import com.openvino.notes.notes.api.port.AttachmentContentConflictException
import com.openvino.notes.notes.api.port.BinarySource
import com.openvino.notes.notes.api.port.RemoteApplyResult
import com.openvino.notes.notes.api.port.RemoteNoteChange
import com.openvino.notes.notes.api.port.RemoteRevision
import com.openvino.notes.notes.api.port.binarySourceOf
import com.openvino.notes.notes.api.port.readAll
import java.time.Instant
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assert.assertArrayEquals
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [35])
class RoomNotesRepositoryTest {
    private val testDispatchers = AppDispatchers(Dispatchers.Unconfined, Dispatchers.Unconfined)

    @Test fun `save preserves the complete note and writes an outbox snapshot`() = runTest {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val database = Room.inMemoryDatabaseBuilder(context, NotesDatabase::class.java).allowMainThreadQueries().build()
        try {
            val repository = RoomNotesRepository(
                database.notesDao(),
                FileAttachmentContentStore(context.cacheDir.resolve("notes-room-save-test"), testDispatchers),
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
                FileAttachmentContentStore(context.cacheDir.resolve("notes-room-remote-test"), testDispatchers),
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
        val content = FileAttachmentContentStore(root, testDispatchers)
        try {
            val repository = RoomNotesRepository(database.notesDao(), content)
            val accountKey = AccountKey("account")
            val noteId = NoteId("note")
            val itemId = ContentItemId("image")
            val firstId = AttachmentId("first")
            val secondId = AttachmentId("second")
            val first = AttachmentMetadata(firstId, noteId, itemId, "first.png", "image/png", 3)
            val second = AttachmentMetadata(secondId, noteId, itemId, "second.png", "image/png", 2)
            content.put(accountKey, first, binarySourceOf(byteArrayOf(1, 2, 3)))
            content.put(accountKey, second, binarySourceOf(byteArrayOf(4, 5)))
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
            val opened = requireNotNull(content.open(accountKey, firstId))
            assertEquals(3L, opened.sizeBytes)
            assertArrayEquals(byteArrayOf(2), opened.read(offset = 1, maxBytes = 1))
            assertArrayEquals(byteArrayOf(1, 2, 3), opened.readAll(maxTotalBytes = 3))

            repository.save(note.copy(attachments = listOf(second)))
            assertEquals(null, content.open(accountKey, firstId))
            repository.delete(accountKey, noteId)
            assertEquals(null, content.open(accountKey, secondId))
        } finally {
            database.close()
            root.deleteRecursively()
        }
    }

    @Test fun `attachment writes consume bounded chunks instead of materializing the source`() = runTest {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val root = context.cacheDir.resolve("notes-room-chunk-test").apply { deleteRecursively() }
        val content = FileAttachmentContentStore(root, testDispatchers)
        val accountKey = AccountKey("account")
        val attachment = AttachmentMetadata(
            AttachmentId("large"),
            NoteId("note"),
            ContentItemId("file"),
            "large.bin",
            "application/octet-stream",
            150_000,
        )
        var largestRequest = 0
        val source = object : BinarySource {
            override val sizeBytes = attachment.sizeBytes

            override suspend fun read(offset: Long, maxBytes: Int): ByteArray {
                largestRequest = maxOf(largestRequest, maxBytes)
                val count = minOf(maxBytes.toLong(), sizeBytes - offset).coerceAtLeast(0).toInt()
                return ByteArray(count) { index -> ((offset + index) % 251).toByte() }
            }
        }
        try {
            content.put(accountKey, attachment, source)

            assertEquals(64 * 1024, largestRequest)
            val opened = requireNotNull(content.open(accountKey, attachment.id))
            assertEquals(attachment.sizeBytes, opened.sizeBytes)
            assertArrayEquals(
                ByteArray(32) { index -> ((70_000 + index) % 251).toByte() },
                opened.read(offset = 70_000, maxBytes = 32),
            )
        } finally {
            root.deleteRecursively()
        }
    }

    @Test fun `attachment id is immutable while identical writes remain idempotent`() = runTest {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val root = context.cacheDir.resolve("notes-room-immutable-attachment-test").apply { deleteRecursively() }
        val content = FileAttachmentContentStore(root, testDispatchers)
        val accountKey = AccountKey("account")
        val attachment = AttachmentMetadata(
            AttachmentId("immutable"),
            NoteId("note"),
            ContentItemId("file"),
            "content.bin",
            "application/octet-stream",
            3,
        )
        try {
            content.put(accountKey, attachment, binarySourceOf(byteArrayOf(1, 2, 3)))
            content.put(accountKey, attachment, binarySourceOf(byteArrayOf(1, 2, 3)))

            try {
                content.put(accountKey, attachment, binarySourceOf(byteArrayOf(3, 2, 1)))
                throw AssertionError("Different content must not replace an existing AttachmentId")
            } catch (_: AttachmentContentConflictException) {
                // Expected: callers must allocate a new AttachmentId for changed bytes.
            }

            assertArrayEquals(
                byteArrayOf(1, 2, 3),
                requireNotNull(content.open(accountKey, attachment.id)).readAll(maxTotalBytes = 3),
            )
        } finally {
            root.deleteRecursively()
        }
    }
    @Test fun `v1 wire format remains readable`() = runTest {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val database = Room.inMemoryDatabaseBuilder(context, NotesDatabase::class.java).allowMainThreadQueries().build()
        try {
            val repository = RoomNotesRepository(
                database.notesDao(),
                FileAttachmentContentStore(context.cacheDir.resolve("notes-room-v1-test"), testDispatchers),
            )
            val accountKey = AccountKey("account")
            val noteId = NoteId("note")
            val note = Note(
                noteId, accountKey, "Title",
                listOf(
                    ContentItem.Text(ContentItemId("body"), "Body"),
                    ContentItem.Image(ContentItemId("image"), AttachmentId("attachment"), "Caption"),
                    ContentItem.File(ContentItemId("file"), AttachmentId("attachment-2")),
                    ContentItem.Link(ContentItemId("link"), "https://example.com", "Example"),
                ),
                listOf(AttachmentMetadata(AttachmentId("attachment"), noteId, ContentItemId("image"), "image.png", "image/png", 123)),
                tags = setOf(NoteTag("important"), NoteTag("work"), NoteTag("v2")),
                isFavorite = true, summary = "Summary", createdAt = Instant.EPOCH, updatedAt = Instant.EPOCH,
            )
            database.notesDao().upsert(
                NoteEntity(
                    accountKey.value, noteId.value, note.title,
                    "text|body|Qm9keQ==\nimage|image|attachment|+Q2FwdGlvbg==\nfile|file|attachment-2\nlink|link|aHR0cHM6Ly9leGFtcGxlLmNvbQ==|+RXhhbXBsZQ==",
                    "attachment|note|image|aW1hZ2UucG5n|aW1hZ2UvcG5n|123",
                    null, "aW1wb3J0YW50\nd29yaw==\ndjI=", true, "Summary", 0L, 0L,
                ),
            )
            assertEquals(note, repository.find(accountKey, noteId))
        } finally {
            database.close()
        }
    }
    @Test
    fun `newly written wire values use v2 marker`() = runTest {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val database = Room.inMemoryDatabaseBuilder(
            context,
            NotesDatabase::class.java,
        ).allowMainThreadQueries().build()

        try {
            val repository = RoomNotesRepository(
                database.notesDao(),
                FileAttachmentContentStore(
                    context.cacheDir.resolve("notes-room-v2-test"),
                    testDispatchers,
                ),
            )

            val accountKey = AccountKey("account")
            val noteId = NoteId("note")

            repository.save(
                Note(
                    id = noteId,
                    accountKey = accountKey,
                    title = "Title",
                    contentItems = listOf(
                        ContentItem.Text(ContentItemId("body"), "Body"),
                    ),
                    attachments = listOf(
                        AttachmentMetadata(
                            AttachmentId("attachment"),
                            noteId,
                            ContentItemId("body"),
                            "file.txt",
                            "text/plain",
                            10,
                        ),
                    ),
                    tags = setOf(NoteTag("work")),
                    isFavorite = false,
                    summary = "Summary",
                    createdAt = Instant.EPOCH,
                    updatedAt = Instant.EPOCH,
                ),
            )

            val entity = database.notesDao().find(
                accountKey.value,
                noteId.value,
            )

            assertTrue(entity!!.contentItems.startsWith("v2:\n"))
            assertTrue(entity.attachments.startsWith("v2:\n"))
            assertTrue(entity.tags.startsWith("v2:\n"))
        } finally {
            database.close()
        }
    }
    @Test fun `v2 rejects malformed UTF-8`() = runTest {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val database = Room.inMemoryDatabaseBuilder(
            context,
            NotesDatabase::class.java,
        ).allowMainThreadQueries().build()

        try {
            val repository = RoomNotesRepository(
                database.notesDao(),
                FileAttachmentContentStore(
                    context.cacheDir.resolve("notes-room-v2-invalid-utf8-test"),
                    testDispatchers,
                ),
            )

            val accountKey = AccountKey("account")
            val noteId = NoteId("note")
            // "wyg" decodes to bytes that are not a valid UTF-8 sequence.
            database.notesDao().upsert(
                NoteEntity(
                    accountKey = accountKey.value,
                    id = noteId.value,
                    title = "Title",
                    contentItems = "v2:\ntext|Ym9keQ|wyg",
                    attachments = "v2:",
                    folderId = null,
                    tags = "v2:",
                    isFavorite = false,
                    summary = null,
                    createdAtMillis = 0L,
                    updatedAtMillis = 0L,
                ),
            )

            try {
                repository.find(accountKey, noteId)
                throw AssertionError("Malformed UTF-8 must be rejected")
            } catch (_: java.nio.charset.CharacterCodingException) {
                // Expected: invalid UTF-8 must not be silently replaced.
            }
        } finally {
            database.close()
        }
    }

    @Test fun `v2 round trips unicode separators and newlines in every field`() = runTest {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val database = Room.inMemoryDatabaseBuilder(
            context,
            NotesDatabase::class.java,
        ).allowMainThreadQueries().build()

        try {
            val repository = RoomNotesRepository(
                database.notesDao(),
                FileAttachmentContentStore(
                    context.cacheDir.resolve("notes-room-v2-unicode-test"),
                    testDispatchers,
                ),
            )

            val accountKey = AccountKey("account")
            val noteId = NoteId("note")
            // Everything that could break the native format: separator, newline,
            // surrogate pair, non-ASCII, control character.
            val nasty = "a|b\nc \u00e9 \uD83D\uDE00 \u0000"
            val nastyId = "id|with\nnewline"

            val note = Note(
                id = noteId,
                accountKey = accountKey,
                title = nasty,
                contentItems = listOf(
                    ContentItem.Text(ContentItemId(nastyId), nasty),
                    ContentItem.Image(
                        ContentItemId("image-$nastyId"),
                        AttachmentId("attachment-$nastyId"),
                        nasty,
                    ),
                    ContentItem.File(
                        ContentItemId("file-$nastyId"),
                        AttachmentId("attachment-2-$nastyId"),
                    ),
                    ContentItem.Link(
                        ContentItemId("link-$nastyId"),
                        "https://example.com/$nastyId?q=$nasty",
                        nasty,
                    ),
                ),
                attachments = listOf(
                    AttachmentMetadata(
                        AttachmentId("attachment-$nastyId"),
                        noteId,
                        ContentItemId("image-$nastyId"),
                        "file|name\n$nasty.txt",
                        "text/plain; charset=$nasty",
                        42L,
                    ),
                ),
                tags = setOf(NoteTag(nasty), NoteTag("tag|with\npipe")),
                isFavorite = false,
                summary = nasty,
                createdAt = Instant.EPOCH,
                updatedAt = Instant.EPOCH,
            )

            repository.save(note)
            assertEquals(note, repository.find(accountKey, noteId))
        } finally {
            database.close()
        }
    }

    @Test fun `v2 distinguishes empty string from null in nullable fields`() = runTest {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val database = Room.inMemoryDatabaseBuilder(
            context,
            NotesDatabase::class.java,
        ).allowMainThreadQueries().build()

        try {
            val repository = RoomNotesRepository(
                database.notesDao(),
                FileAttachmentContentStore(
                    context.cacheDir.resolve("notes-room-v2-empty-string-test"),
                    testDispatchers,
                ),
            )

            val accountKey = AccountKey("account")
            val noteId = NoteId("note")

            val note = Note(
                id = noteId,
                accountKey = accountKey,
                title = "Title",
                contentItems = listOf(
                    ContentItem.Image(
                        ContentItemId("image-null"),
                        AttachmentId("attachment-1"),
                        null,
                    ),
                    ContentItem.Image(
                        ContentItemId("image-empty"),
                        AttachmentId("attachment-2"),
                        "",
                    ),
                    ContentItem.Link(
                        ContentItemId("link-null"),
                        "https://example.com",
                        null,
                    ),
                    ContentItem.Link(
                        ContentItemId("link-empty"),
                        "https://example.com",
                        "",
                    ),
                ),
                tags = emptySet(),
                isFavorite = false,
                summary = null,
                createdAt = Instant.EPOCH,
                updatedAt = Instant.EPOCH,
            )

            repository.save(note)
            val restored = repository.find(accountKey, noteId)!!

            // The difference between null and "" must survive a round trip.
            assertEquals(null, (restored.contentItems[0] as ContentItem.Image).caption)
            assertEquals("", (restored.contentItems[1] as ContentItem.Image).caption)
            assertEquals(null, (restored.contentItems[2] as ContentItem.Link).label)
            assertEquals("", (restored.contentItems[3] as ContentItem.Link).label)
        } finally {
            database.close()
        }
    }

    @Test fun `v2 marker with trailing newline decodes to empty collections`() = runTest {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val database = Room.inMemoryDatabaseBuilder(
            context,
            NotesDatabase::class.java,
        ).allowMainThreadQueries().build()

        try {
            val repository = RoomNotesRepository(
                database.notesDao(),
                FileAttachmentContentStore(
                    context.cacheDir.resolve("notes-room-v2-empty-test"),
                    testDispatchers,
                ),
            )

            val accountKey = AccountKey("account")
            val noteId = NoteId("note")
            // Regression: "v2:\n" used to crash content/attachments decoding
            // and to produce {NoteTag("")} for tags. It must now yield empty
            // collections in all three cases.
            database.notesDao().upsert(
                NoteEntity(
                    accountKey = accountKey.value,
                    id = noteId.value,
                    title = "Title",
                    contentItems = "v2:\n",
                    attachments = "v2:\n",
                    folderId = null,
                    tags = "v2:\n",
                    isFavorite = false,
                    summary = null,
                    createdAtMillis = 0L,
                    updatedAtMillis = 0L,
                ),
            )

            val note = repository.find(accountKey, noteId)!!
            assertEquals(emptyList<ContentItem>(), note.contentItems)
            assertEquals(emptyList<AttachmentMetadata>(), note.attachments)
            assertEquals(emptySet<NoteTag>(), note.tags)
        } finally {
            database.close()
        }
    }

    @Test fun `v2 survives database close and reopen`() = runTest {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val databaseFile = context.getDatabasePath("reopen-test.db")
        databaseFile.delete()

        // First open: write a note with nasty identifiers.
        var database = Room.databaseBuilder(context, NotesDatabase::class.java, "reopen-test.db")
            .allowMainThreadQueries()
            .build()
        try {
            val repository = RoomNotesRepository(
                database.notesDao(),
                FileAttachmentContentStore(context.cacheDir.resolve("notes-room-reopen-test"), testDispatchers),
            )
            val note = Note(
                id = NoteId("note|with\nnewline"),
                accountKey = AccountKey("account"),
                title = "Title|with\nseparator",
                contentItems = listOf(ContentItem.Text(ContentItemId("body|id"), "Body")),
                tags = setOf(NoteTag("tag|with\npipe")),
                isFavorite = false,
                summary = null,
                createdAt = Instant.EPOCH,
                updatedAt = Instant.EPOCH,
            )
            repository.save(note)
        } finally {
            database.close()
        }

        // Second open: verify the note is still readable and intact.
        database = Room.databaseBuilder(context, NotesDatabase::class.java, "reopen-test.db")
            .allowMainThreadQueries()
            .build()
        try {
            val repository = RoomNotesRepository(
                database.notesDao(),
                FileAttachmentContentStore(context.cacheDir.resolve("notes-room-reopen-test"), testDispatchers),
            )
            val restored = repository.find(AccountKey("account"), NoteId("note|with\nnewline"))
            assertEquals("Title|with\nseparator", restored?.title)
            assertEquals(setOf(NoteTag("tag|with\npipe")), restored?.tags)
        } finally {
            database.close()
            databaseFile.delete()
        }
    }

    @Test fun `v2 rejects empty line between records`() = runTest {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val database = Room.inMemoryDatabaseBuilder(context, NotesDatabase::class.java)
            .allowMainThreadQueries().build()

        try {
            val repository = RoomNotesRepository(
                database.notesDao(),
                FileAttachmentContentStore(
                    context.cacheDir.resolve("notes-room-v2-empty-line-test"),
                    testDispatchers,
                ),
            )

            val accountKey = AccountKey("account")
            val noteId = NoteId("note")
            database.notesDao().upsert(
                NoteEntity(
                    accountKey = accountKey.value,
                    id = noteId.value,
                    title = "Title",
                    contentItems = "v2:\ntext|Ym9keQ|Ym9keQ\n\ntext|Ym9keQ|Ym9keQ",
                    attachments = "v2:",
                    folderId = null,
                    tags = "v2:",
                    isFavorite = false,
                    summary = null,
                    createdAtMillis = 0L,
                    updatedAtMillis = 0L,
                ),
            )

            try {
                repository.find(accountKey, noteId)
                throw AssertionError("Empty line between records must be rejected")
            } catch (_: IllegalArgumentException) {
                // Expected.
            }
        } finally {
            database.close()
        }
    }
}
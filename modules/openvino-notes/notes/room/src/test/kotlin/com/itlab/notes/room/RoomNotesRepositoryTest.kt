// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package com.itlab.notes.room

import android.content.Context
import androidx.room.Room
import androidx.test.core.app.ApplicationProvider
import com.itlab.identity.api.AccountId
import com.itlab.notes.api.Note
import com.itlab.notes.api.NoteId
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
    @Test fun `save writes note and sync outbox in one repository operation`() = runTest {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val database = Room.inMemoryDatabaseBuilder(context, NotesDatabase::class.java).allowMainThreadQueries().build()
        try {
            val repository = RoomNotesRepository(database.notesDao())
            val accountId = AccountId("account")
            val note = Note(NoteId("note"), accountId, "Title", "Body", updatedAt = Instant.EPOCH)

            repository.save(note)

            assertEquals(note, repository.find(accountId, note.id))
            assertEquals(listOf(note.id), repository.pendingChanges(accountId).map { it.noteId })
        } finally {
            database.close()
        }
    }
}

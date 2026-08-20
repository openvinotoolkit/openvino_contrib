// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package com.openvino.notes.notes.room

import android.content.Context
import androidx.room.Room
import androidx.test.core.app.ApplicationProvider
import com.openvino.notes.kernel.AccountKey
import com.openvino.notes.notes.api.Folder
import com.openvino.notes.notes.api.FolderId
import com.openvino.notes.notes.api.port.FolderRemoteApplyResult
import com.openvino.notes.notes.api.port.RemoteFolderChange
import com.openvino.notes.notes.api.port.RemoteRevision
import java.time.Instant
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [35])
class RoomFolderRepositoryTest {
    @Test fun `local folder lifecycle writes outbox changes`() = runTest {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val database = Room.inMemoryDatabaseBuilder(context, NotesDatabase::class.java).allowMainThreadQueries().build()
        try {
            val repository = RoomFolderRepository(database.folderDao())
            val folder = folder()

            repository.save(folder)

            assertEquals(listOf(folder), repository.observe(folder.accountKey).first())
            assertEquals(folder, repository.findByName(folder.accountKey, "PROJECTS"))
            assertEquals(listOf(folder), repository.pendingFolderChanges(folder.accountKey).map { it.payload })
            assertEquals(true, repository.delete(folder.accountKey, folder.id))
            assertEquals(null, repository.find(folder.accountKey, folder.id))
            assertEquals(2, repository.pendingFolderChanges(folder.accountKey).size)
        } finally {
            database.close()
        }
    }

    @Test fun `remote folder changes apply without creating local outbox entries`() = runTest {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val database = Room.inMemoryDatabaseBuilder(context, NotesDatabase::class.java).allowMainThreadQueries().build()
        try {
            val repository = RoomFolderRepository(database.folderDao())
            val folder = folder()
            val revision1 = RemoteRevision("1")
            val revision2 = RemoteRevision("2")

            assertEquals(
                listOf(FolderRemoteApplyResult.Applied(folder.id, revision1)),
                repository.applyRemoteFolders(
                    folder.accountKey,
                    listOf(RemoteFolderChange.Upsert(folder, null, revision1)),
                ),
            )
            assertEquals(emptyList<FolderId>(), repository.pendingFolderChanges(folder.accountKey).map { it.folderId })
            assertEquals(
                listOf(FolderRemoteApplyResult.TombstoneApplied(folder.id, revision2)),
                repository.applyRemoteFolders(
                    folder.accountKey,
                    listOf(RemoteFolderChange.Tombstone(folder.id, revision2)),
                ),
            )
        } finally {
            database.close()
        }
    }

    private fun folder() = Folder(
        id = FolderId("folder"),
        accountKey = AccountKey("account"),
        name = "Projects",
        createdAt = Instant.EPOCH,
        updatedAt = Instant.EPOCH,
    )
}

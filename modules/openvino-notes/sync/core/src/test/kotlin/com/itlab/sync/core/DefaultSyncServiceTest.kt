package com.itlab.sync.core

import com.itlab.cloud.api.FakeRemoteStore
import com.itlab.identity.api.AccountId
import com.itlab.identity.api.FakeIdentityService
import com.itlab.identity.api.UserSession
import com.itlab.kernel.NoOpAppLogger
import com.itlab.notes.api.FakeNotesSyncPort
import com.itlab.notes.api.LocalChangeKind
import com.itlab.notes.api.LocalNoteChange
import com.itlab.notes.api.Note
import com.itlab.notes.api.NoteId
import com.itlab.sync.api.FakeSyncStateStore
import com.itlab.sync.api.SyncOutcome
import com.itlab.sync.api.SyncReason
import java.time.Instant
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class DefaultSyncServiceTest {
    @Test fun `uploads and acknowledges local changes`() = runTest {
        val account = AccountId("account")
        val identity = FakeIdentityService().apply { setSession(UserSession(account, "User", "u@example.test")) }
        val note = Note(NoteId("note"), account, "Title", "Body", updatedAt = Instant.EPOCH)
        val notes = FakeNotesSyncPort().apply { pending += LocalNoteChange("change", account, note.id, LocalChangeKind.UPSERT, Instant.EPOCH, note) }
        val remote = FakeRemoteStore()
        val service = DefaultSyncService(identity, notes, remote, remote, FakeSyncStateStore(), { Instant.EPOCH }, NoOpAppLogger)

        assertEquals(SyncOutcome.Completed(1, 0), service.sync(SyncReason.USER))
        assertTrue(notes.pending.isEmpty())
        assertTrue(remote.objects.isNotEmpty())
    }
}

// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package com.itlab.notes.core

import com.itlab.identity.api.AccountId
import com.itlab.identity.api.FakeIdentityService
import com.itlab.identity.api.UserSession
import com.itlab.notes.api.FakeNotesRepository
import com.itlab.notes.api.NoteDraft
import com.itlab.notes.api.NoteId
import com.itlab.notes.api.NoteMutationOutcome
import java.time.Instant
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class DefaultNotesServiceTest {
    @Test fun `create requires a session`() = runTest {
        val service = DefaultNotesService(FakeIdentityService(), FakeNotesRepository(), { Instant.EPOCH })
        assertEquals(NoteMutationOutcome.SignedOut, service.create(NoteDraft("Title", "Body")))
    }

    @Test fun `create trims title and persists note`() = runTest {
        val identity = FakeIdentityService().apply { setSession(UserSession(AccountId("account"), "User", "u@example.test")) }
        val repository = FakeNotesRepository()
        val service = DefaultNotesService(identity, repository, { Instant.EPOCH }) { NoteId("note-1") }
        val outcome = service.create(NoteDraft("  Title  ", "Body"))
        assertTrue(outcome is NoteMutationOutcome.Saved)
        assertEquals("Title", repository.find(AccountId("account"), NoteId("note-1"))?.title)
    }
}

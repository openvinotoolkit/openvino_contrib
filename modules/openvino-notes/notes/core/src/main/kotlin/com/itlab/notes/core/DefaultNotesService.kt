// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package com.itlab.notes.core

import com.itlab.identity.api.AuthenticationState
import com.itlab.identity.api.SessionReader
import com.itlab.kernel.AppClock
import com.itlab.notes.api.Note
import com.itlab.notes.api.NoteDraft
import com.itlab.notes.api.NoteId
import com.itlab.notes.api.NoteMutationOutcome
import com.itlab.notes.api.NotesRepository
import com.itlab.notes.api.NotesService
import com.itlab.notes.api.UpdateNoteCommand
import java.util.UUID
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flatMapLatest
import kotlinx.coroutines.flow.flowOf
import kotlinx.coroutines.ExperimentalCoroutinesApi

@OptIn(ExperimentalCoroutinesApi::class)
class DefaultNotesService(
    private val sessions: SessionReader,
    private val repository: NotesRepository,
    private val clock: AppClock,
    private val newId: () -> NoteId = { NoteId(UUID.randomUUID().toString()) },
) : NotesService {
    override fun observeNotes(): Flow<List<Note>> = sessions.authenticationState.flatMapLatest { state ->
        when (state) {
            AuthenticationState.SignedOut -> flowOf(emptyList())
            is AuthenticationState.SignedIn -> repository.observe(state.session.accountId)
        }
    }

    override suspend fun create(draft: NoteDraft): NoteMutationOutcome {
        val accountId = sessions.currentAccountId() ?: return NoteMutationOutcome.SignedOut
        validate(draft.title, draft.body)?.let { return it }
        val note = Note(newId(), accountId, draft.title.trim(), draft.body, draft.folderId, clock.now())
        repository.save(note)
        return NoteMutationOutcome.Saved(note)
    }

    override suspend fun update(command: UpdateNoteCommand): NoteMutationOutcome {
        val accountId = sessions.currentAccountId() ?: return NoteMutationOutcome.SignedOut
        validate(command.title, command.body)?.let { return it }
        repository.find(accountId, command.id) ?: return NoteMutationOutcome.NotFound
        val note = Note(command.id, accountId, command.title.trim(), command.body, command.folderId, clock.now())
        repository.save(note)
        return NoteMutationOutcome.Saved(note)
    }

    override suspend fun delete(id: NoteId): NoteMutationOutcome {
        val accountId = sessions.currentAccountId() ?: return NoteMutationOutcome.SignedOut
        return if (repository.delete(accountId, id)) NoteMutationOutcome.Deleted else NoteMutationOutcome.NotFound
    }

    private fun validate(title: String, body: String): NoteMutationOutcome.Invalid? = when {
        title.isBlank() -> NoteMutationOutcome.Invalid("title", "must not be blank")
        title.length > 200 -> NoteMutationOutcome.Invalid("title", "must not exceed 200 characters")
        body.length > 100_000 -> NoteMutationOutcome.Invalid("body", "must not exceed 100000 characters")
        else -> null
    }
}

data class NotesCoreComponent(val notesService: NotesService) {
    companion object {
        fun create(sessions: SessionReader, repository: NotesRepository, clock: AppClock): NotesCoreComponent =
            NotesCoreComponent(DefaultNotesService(sessions, repository, clock))
    }
}

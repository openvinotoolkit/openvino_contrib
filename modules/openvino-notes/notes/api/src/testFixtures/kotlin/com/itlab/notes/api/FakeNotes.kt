// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package com.itlab.notes.api

import com.itlab.identity.api.AccountId
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.map

class FakeNotesRepository : NotesRepository {
    private val notes = MutableStateFlow<List<Note>>(emptyList())
    override fun observe(accountId: AccountId): Flow<List<Note>> = notes.map { values -> values.filter { it.accountId == accountId } }
    override suspend fun find(accountId: AccountId, id: NoteId): Note? = notes.value.firstOrNull { it.accountId == accountId && it.id == id }
    override suspend fun save(note: Note) { notes.value = notes.value.filterNot { it.id == note.id } + note }
    override suspend fun delete(accountId: AccountId, id: NoteId): Boolean {
        val before = notes.value.size
        notes.value = notes.value.filterNot { it.accountId == accountId && it.id == id }
        return before != notes.value.size
    }
}

class FakeNotesSyncPort : NotesSyncPort {
    val pending = mutableListOf<LocalNoteChange>()
    val appliedRemote = mutableListOf<RemoteNoteChange>()
    override suspend fun pendingChanges(accountId: AccountId, limit: Int): List<LocalNoteChange> = pending.filter { it.accountId == accountId }.take(limit)
    override suspend fun acknowledge(changeIds: Set<String>) { pending.removeAll { it.changeId in changeIds } }
    override suspend fun applyRemote(accountId: AccountId, changes: List<RemoteNoteChange>) { appliedRemote += changes }
}

class FakeNotesService(initial: List<Note> = emptyList()) : NotesService {
    private val notes = MutableStateFlow(initial)
    override fun observeNotes(): Flow<List<Note>> = notes
    override suspend fun create(draft: NoteDraft): NoteMutationOutcome = NoteMutationOutcome.Invalid("fake", "configure outcome in test")
    override suspend fun update(command: UpdateNoteCommand): NoteMutationOutcome = NoteMutationOutcome.Invalid("fake", "configure outcome in test")
    override suspend fun delete(id: NoteId): NoteMutationOutcome = NoteMutationOutcome.NotFound
    fun emit(values: List<Note>) { notes.value = values }
}

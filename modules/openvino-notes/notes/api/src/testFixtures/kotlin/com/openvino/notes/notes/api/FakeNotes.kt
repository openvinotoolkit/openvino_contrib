// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package com.openvino.notes.notes.api

import com.openvino.notes.kernel.AccountKey
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.map

class FakeNotesRepository : NotesRepository {
    private val notes = MutableStateFlow<List<Note>>(emptyList())
    override fun observe(accountKey: AccountKey): Flow<List<Note>> = notes.map { values -> values.filter { it.accountKey == accountKey } }
    override suspend fun find(accountKey: AccountKey, id: NoteId): Note? = notes.value.firstOrNull { it.accountKey == accountKey && it.id == id }
    override suspend fun save(note: Note) { notes.value = notes.value.filterNot { it.accountKey == note.accountKey && it.id == note.id } + note }
    override suspend fun delete(accountKey: AccountKey, id: NoteId): Boolean {
        val before = notes.value.size
        notes.value = notes.value.filterNot { it.accountKey == accountKey && it.id == id }
        return before != notes.value.size
    }
}

class FakeNotesSyncPort : NotesSyncPort {
    val events = mutableListOf<String>()
    val pending = mutableListOf<LocalNoteChange>()
    val appliedRemote = mutableListOf<RemoteNoteChange>()
    var remoteResults: List<RemoteApplyResult> = emptyList()
    override suspend fun pendingChanges(accountKey: AccountKey, limit: Int): List<LocalNoteChange> {
        events += "pending"
        return pending.filter { it.accountKey == accountKey }.take(limit)
    }
    override suspend fun acknowledge(accountKey: AccountKey, changeIds: Set<String>) {
        events += "acknowledge"
        pending.removeAll { it.accountKey == accountKey && it.changeId in changeIds }
    }
    override suspend fun applyRemote(accountKey: AccountKey, changes: List<RemoteNoteChange>): List<RemoteApplyResult> {
        events += "applyRemote"
        appliedRemote += changes
        return remoteResults
    }
}

class FakeNotesService(initial: List<Note> = emptyList()) : NotesService {
    private val notes = MutableStateFlow(initial)
    override fun observeNotes(): Flow<List<Note>> = notes
    override suspend fun get(id: NoteId): Note? = notes.value.firstOrNull { it.id == id }
    override suspend fun create(draft: NoteDraft): NoteMutationOutcome = NoteMutationOutcome.Invalid("fake", "create outcome not configured")
    override suspend fun update(command: UpdateNoteCommand): NoteMutationOutcome {
        val current = get(command.id) ?: return NoteMutationOutcome.NotFound
        val updated = current.copy(
            title = command.title.resolve(current.title),
            contentItems = command.contentItems.resolve(current.contentItems),
            attachments = command.attachments.resolve(current.attachments),
            folderId = command.folderId.resolve(current.folderId),
            tags = command.tags.resolve(current.tags),
            isFavorite = command.isFavorite.resolve(current.isFavorite),
            summary = command.summary.resolve(current.summary),
        )
        notes.value = notes.value.map { if (it.id == updated.id) updated else it }
        return NoteMutationOutcome.Saved(updated)
    }
    override suspend fun delete(id: NoteId): NoteMutationOutcome {
        val existed = notes.value.any { it.id == id }
        notes.value = notes.value.filterNot { it.id == id }
        return if (existed) NoteMutationOutcome.Deleted else NoteMutationOutcome.NotFound
    }
    fun emit(values: List<Note>) { notes.value = values }
}

@Suppress("UNCHECKED_CAST")
private fun <T> FieldUpdate<T>.resolve(current: T): T = when (this) {
    FieldUpdate.Keep -> current
    is FieldUpdate.Set -> value
}

// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package com.openvino.notes.notes.api

import com.openvino.notes.kernel.AccountKey
import com.openvino.notes.notes.api.port.AttachmentContentConflictException
import com.openvino.notes.notes.api.port.AttachmentContentPort
import com.openvino.notes.notes.api.port.BinarySource
import com.openvino.notes.notes.api.port.FolderRemoteApplyResult
import com.openvino.notes.notes.api.port.FolderRepository
import com.openvino.notes.notes.api.port.FolderSyncPort
import com.openvino.notes.notes.api.port.LocalFolderChange
import com.openvino.notes.notes.api.port.LocalNoteChange
import com.openvino.notes.notes.api.port.NotesRepository
import com.openvino.notes.notes.api.port.NotesSyncPort
import com.openvino.notes.notes.api.port.RemoteApplyResult
import com.openvino.notes.notes.api.port.RemoteFolderChange
import com.openvino.notes.notes.api.port.RemoteNoteChange
import com.openvino.notes.notes.api.port.binarySourceOf
import com.openvino.notes.notes.api.port.readAll
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.map

class FakeNotesRepository : NotesRepository {
    private val notes = MutableStateFlow<List<Note>>(emptyList())

    override fun observe(accountKey: AccountKey): Flow<List<Note>> = notes.map { values -> values.filter { it.accountKey == accountKey } }

    override suspend fun find(
        accountKey: AccountKey,
        id: NoteId,
    ): Note? =
        notes.value.firstOrNull {
            it.accountKey == accountKey &&
                it.id == id
        }

    override suspend fun countInFolder(
        accountKey: AccountKey,
        folderId: FolderId,
    ): Int = notes.value.count { it.accountKey == accountKey && it.folderId == folderId }

    override suspend fun save(note: Note) {
        notes.value =
            notes.value.filterNot { it.accountKey == note.accountKey && it.id == note.id } + note
    }

    override suspend fun delete(
        accountKey: AccountKey,
        id: NoteId,
    ): Boolean {
        val before = notes.value.size
        notes.value = notes.value.filterNot { it.accountKey == accountKey && it.id == id }
        return before != notes.value.size
    }
}

class FakeFolderRepository : FolderRepository {
    private val folders = MutableStateFlow<List<Folder>>(emptyList())

    override fun observe(accountKey: AccountKey): Flow<List<Folder>> =
        folders.map { values -> values.filter { it.accountKey == accountKey } }

    override suspend fun find(
        accountKey: AccountKey,
        id: FolderId,
    ): Folder? = folders.value.firstOrNull { it.accountKey == accountKey && it.id == id }

    override suspend fun findByName(
        accountKey: AccountKey,
        name: String,
    ): Folder? = folders.value.firstOrNull { it.accountKey == accountKey && it.name.equals(name, ignoreCase = true) }

    override suspend fun save(folder: Folder) {
        folders.value = folders.value.filterNot { it.accountKey == folder.accountKey && it.id == folder.id } + folder
    }

    override suspend fun delete(
        accountKey: AccountKey,
        id: FolderId,
    ): Boolean {
        val before = folders.value.size
        folders.value = folders.value.filterNot { it.accountKey == accountKey && it.id == id }
        return before != folders.value.size
    }
}

class FakeNotesSyncPort : NotesSyncPort {
    val events = mutableListOf<String>()
    val pending = mutableListOf<LocalNoteChange>()
    val appliedRemote = mutableListOf<RemoteNoteChange>()
    var remoteResults: List<RemoteApplyResult> = emptyList()

    override suspend fun pendingChanges(
        accountKey: AccountKey,
        limit: Int,
    ): List<LocalNoteChange> {
        events += "pending"
        return pending.filter { it.accountKey == accountKey }.take(limit)
    }

    override suspend fun acknowledge(
        accountKey: AccountKey,
        changeIds: Set<String>,
    ) {
        events += "acknowledge"
        pending.removeAll { it.accountKey == accountKey && it.changeId in changeIds }
    }

    override suspend fun applyRemote(
        accountKey: AccountKey,
        changes: List<RemoteNoteChange>,
    ): List<RemoteApplyResult> {
        events += "applyRemote"
        appliedRemote += changes
        return remoteResults
    }
}

class FakeFolderSyncPort : FolderSyncPort {
    val pending = mutableListOf<LocalFolderChange>()
    val appliedRemote = mutableListOf<RemoteFolderChange>()
    var remoteResults: List<FolderRemoteApplyResult> = emptyList()

    override suspend fun pendingFolderChanges(
        accountKey: AccountKey,
        limit: Int,
    ): List<LocalFolderChange> = pending.filter { it.accountKey == accountKey }.take(limit)

    override suspend fun acknowledgeFolderChanges(
        accountKey: AccountKey,
        changeIds: Set<String>,
    ) {
        pending.removeAll { it.accountKey == accountKey && it.changeId in changeIds }
    }

    override suspend fun applyRemoteFolders(
        accountKey: AccountKey,
        changes: List<RemoteFolderChange>,
    ): List<FolderRemoteApplyResult> {
        appliedRemote += changes
        return remoteResults
    }
}

class FakeAttachmentContentPort : AttachmentContentPort {
    private val content = mutableMapOf<Pair<AccountKey, AttachmentId>, ByteArray>()

    override suspend fun put(
        accountKey: AccountKey,
        attachment: AttachmentMetadata,
        source: BinarySource,
    ) {
        require(source.sizeBytes == attachment.sizeBytes)
        val key = accountKey to attachment.id
        val bytes = source.readAll(attachment.sizeBytes)
        val existing = content[key]
        if (existing != null && !existing.contentEquals(bytes)) {
            throw AttachmentContentConflictException(attachment.id)
        }
        content.putIfAbsent(key, bytes)
    }

    override suspend fun open(
        accountKey: AccountKey,
        attachmentId: AttachmentId,
    ): BinarySource? = content[accountKey to attachmentId]?.let(::binarySourceOf)

    override suspend fun delete(
        accountKey: AccountKey,
        attachmentId: AttachmentId,
    ): Boolean = content.remove(accountKey to attachmentId) != null
}

class FakeNotesService(
    initial: List<Note> = emptyList(),
) : NotesService {
    private val notes = MutableStateFlow(initial)
    var createOutcome: NoteMutationOutcome? = null
    val currentNotes: List<Note> get() = notes.value

    override fun observeNotes(): Flow<List<Note>> = notes

    override suspend fun get(id: NoteId): Note? = notes.value.firstOrNull { it.id == id }
    override suspend fun create(draft: NoteDraft): NoteMutationOutcome {
        val configured = createOutcome
        if (configured != null) return configured
        val created = Note(
            id = NoteId("fake-note-${notes.value.size + 1}"),
            accountKey = AccountKey("fake-account"),
            title = draft.title,
            contentItems = draft.contentItems,
            attachments = draft.attachments,
            folderId = draft.folderId,
            tags = draft.tags,
            isFavorite = draft.isFavorite,
            createdAt = java.time.Instant.EPOCH,
            updatedAt = java.time.Instant.EPOCH,
        )
        notes.value = notes.value + created
        return NoteMutationOutcome.Saved(created)
    }
    override suspend fun update(command: UpdateNoteCommand): NoteMutationOutcome {
        val current = get(command.id) ?: return NoteMutationOutcome.NotFound
        val updated =
            current.copy(
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

    fun emit(values: List<Note>) {
        notes.value = values
    }
}

class FakeFolderService(initial: List<Folder> = emptyList()) : FolderService {
    private val folders = MutableStateFlow(initial)
    override fun observeFolders(): Flow<List<Folder>> = folders
    override suspend fun getFolder(id: FolderId): Folder? = folders.value.firstOrNull { it.id == id }
    override suspend fun createFolder(draft: FolderDraft): FolderMutationOutcome {
        val created = Folder(
            id = FolderId("fake-folder-${folders.value.size + 1}"),
            accountKey = AccountKey("fake-account"),
            name = draft.name,
            createdAt = java.time.Instant.EPOCH,
            updatedAt = java.time.Instant.EPOCH,
        )
        folders.value = folders.value + created
        return FolderMutationOutcome.Saved(created)
    }
    override suspend fun renameFolder(id: FolderId, name: String): FolderMutationOutcome {
        val current = getFolder(id) ?: return FolderMutationOutcome.NotFound
        val updated = current.copy(name = name)
        folders.value = folders.value.map { if (it.id == id) updated else it }
        return FolderMutationOutcome.Saved(updated)
    }
    override suspend fun deleteFolder(id: FolderId): FolderMutationOutcome {
        val existed = folders.value.any { it.id == id }
        folders.value = folders.value.filterNot { it.id == id }
        return if (existed) FolderMutationOutcome.Deleted else FolderMutationOutcome.NotFound
    }
    override suspend fun moveNote(noteId: NoteId, folderId: FolderId?): MoveNoteOutcome = MoveNoteOutcome.Invalid("not configured")
}

@Suppress("UNCHECKED_CAST")
private fun <T> FieldUpdate<T>.resolve(current: T): T =
    when (this) {
        FieldUpdate.Keep -> current
        is FieldUpdate.Set -> value
    }

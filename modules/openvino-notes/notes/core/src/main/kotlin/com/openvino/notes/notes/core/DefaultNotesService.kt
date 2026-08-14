// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package com.openvino.notes.notes.core

import com.openvino.notes.kernel.AccountKey
import com.openvino.notes.kernel.AccountScope
import com.openvino.notes.kernel.AppClock
import com.openvino.notes.notes.api.AttachmentMetadata
import com.openvino.notes.notes.api.ContentItem
import com.openvino.notes.notes.api.FieldUpdate
import com.openvino.notes.notes.api.Folder
import com.openvino.notes.notes.api.FolderDraft
import com.openvino.notes.notes.api.FolderId
import com.openvino.notes.notes.api.FolderMutationOutcome
import com.openvino.notes.notes.api.FolderService
import com.openvino.notes.notes.api.MoveNoteOutcome
import com.openvino.notes.notes.api.Note
import com.openvino.notes.notes.api.NoteDraft
import com.openvino.notes.notes.api.NoteId
import com.openvino.notes.notes.api.NoteMutationOutcome
import com.openvino.notes.notes.api.NotesService
import com.openvino.notes.notes.api.UpdateNoteCommand
import com.openvino.notes.notes.api.port.FolderRepository
import com.openvino.notes.notes.api.port.NotesRepository
import java.util.UUID
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flatMapLatest
import kotlinx.coroutines.flow.flowOf

@OptIn(ExperimentalCoroutinesApi::class)
class DefaultNotesService(
    private val accounts: AccountScope,
    private val repository: NotesRepository,
    private val folders: FolderRepository,
    private val clock: AppClock,
    private val newId: () -> NoteId = { NoteId(UUID.randomUUID().toString()) },
) : NotesService {
    override fun observeNotes(): Flow<List<Note>> = accounts.activeAccountKey.flatMapLatest { accountKey ->
        accountKey?.let(repository::observe) ?: flowOf(emptyList())
    }

    override suspend fun get(id: NoteId): Note? =
        accounts.currentAccountKey()?.let { repository.find(it, id) }

    override suspend fun create(draft: NoteDraft): NoteMutationOutcome {
        val accountKey = accounts.currentAccountKey() ?: return NoteMutationOutcome.SignedOut
        val noteId = newId()
        val now = clock.now()
        val note = Note(
            id = noteId,
            accountKey = accountKey,
            title = draft.title.trim(),
            contentItems = draft.contentItems,
            attachments = draft.attachments.normalizeNoteId(noteId),
            folderId = draft.folderId,
            tags = draft.tags,
            isFavorite = draft.isFavorite,
            summary = draft.summary,
            createdAt = now,
            updatedAt = now,
        )
        validate(note)?.let { return it }
        validateFolder(accountKey, note.folderId)?.let { return it }
        repository.save(note)
        return NoteMutationOutcome.Saved(note)
    }

    override suspend fun update(command: UpdateNoteCommand): NoteMutationOutcome {
        val accountKey = accounts.currentAccountKey() ?: return NoteMutationOutcome.SignedOut
        val existing = repository.find(accountKey, command.id) ?: return NoteMutationOutcome.NotFound
        val candidate = existing.copy(
            title = command.title.resolve(existing.title).trim(),
            contentItems = command.contentItems.resolve(existing.contentItems),
            attachments = command.attachments.resolve(existing.attachments).normalizeNoteId(existing.id),
            folderId = command.folderId.resolve(existing.folderId),
            tags = command.tags.resolve(existing.tags),
            isFavorite = command.isFavorite.resolve(existing.isFavorite),
            summary = command.summary.resolve(existing.summary),
        )
        validate(candidate)?.let { return it }
        validateFolder(accountKey, candidate.folderId)?.let { return it }
        if (candidate == existing) return NoteMutationOutcome.Saved(existing)

        val updated = candidate.copy(updatedAt = clock.now())
        repository.save(updated)
        return NoteMutationOutcome.Saved(updated)
    }

    override suspend fun delete(id: NoteId): NoteMutationOutcome {
        val accountKey = accounts.currentAccountKey() ?: return NoteMutationOutcome.SignedOut
        return if (repository.delete(accountKey, id)) NoteMutationOutcome.Deleted else NoteMutationOutcome.NotFound
    }

    private fun validate(note: Note): NoteMutationOutcome.Invalid? {
        if (note.title.isBlank()) return invalid("title", "must not be blank")
        if (note.title.length > MAX_TITLE_LENGTH) return invalid("title", "must not exceed $MAX_TITLE_LENGTH characters")
        if (note.contentItems.filterIsInstance<ContentItem.Text>().sumOf { it.text.length } > MAX_TEXT_LENGTH) {
            return invalid("contentItems", "text must not exceed $MAX_TEXT_LENGTH characters")
        }
        if (note.contentItems.map(ContentItem::id).toSet().size != note.contentItems.size) {
            return invalid("contentItems", "ids must be unique")
        }
        if (note.attachments.map(AttachmentMetadata::id).toSet().size != note.attachments.size) {
            return invalid("attachments", "ids must be unique")
        }
        val contentIds = note.contentItems.map(ContentItem::id).toSet()
        if (note.attachments.any { it.noteId != note.id || it.contentItemId !in contentIds || it.sizeBytes < 0 }) {
            return invalid("attachments", "must reference this note and an existing content item")
        }
        val attachmentIds = note.attachments.map(AttachmentMetadata::id).toSet()
        val referencedAttachmentIds = note.contentItems.mapNotNull {
            when (it) {
                is ContentItem.Image -> it.attachmentId
                is ContentItem.File -> it.attachmentId
                else -> null
            }
        }
        if (referencedAttachmentIds.any { it !in attachmentIds }) {
            return invalid("contentItems", "attachment references must resolve")
        }
        return null
    }

    private fun invalid(field: String, reason: String) = NoteMutationOutcome.Invalid(field, reason)

    private suspend fun validateFolder(accountKey: AccountKey, folderId: FolderId?): NoteMutationOutcome.Invalid? =
        if (folderId != null && folders.find(accountKey, folderId) == null) {
            invalid("folderId", "folder does not exist")
        } else {
            null
        }

    private companion object {
        const val MAX_TITLE_LENGTH = 200
        const val MAX_TEXT_LENGTH = 100_000
    }
}

@OptIn(ExperimentalCoroutinesApi::class)
class DefaultFolderService(
    private val accounts: AccountScope,
    private val folders: FolderRepository,
    private val notesRepository: NotesRepository,
    private val notesService: NotesService,
    private val clock: AppClock,
    private val newId: () -> FolderId = { FolderId(UUID.randomUUID().toString()) },
) : FolderService {
    override fun observeFolders(): Flow<List<Folder>> = accounts.activeAccountKey.flatMapLatest { accountKey ->
        accountKey?.let(folders::observe) ?: flowOf(emptyList())
    }

    override suspend fun getFolder(id: FolderId): Folder? =
        accounts.currentAccountKey()?.let { folders.find(it, id) }

    override suspend fun createFolder(draft: FolderDraft): FolderMutationOutcome {
        val accountKey = accounts.currentAccountKey() ?: return FolderMutationOutcome.SignedOut
        val name = draft.name.trim()
        validateName(name)?.let { return it }
        if (folders.findByName(accountKey, name) != null) return duplicateName()
        val now = clock.now()
        val folder = Folder(newId(), accountKey, name, now, now)
        folders.save(folder)
        return FolderMutationOutcome.Saved(folder)
    }

    override suspend fun renameFolder(id: FolderId, name: String): FolderMutationOutcome {
        val accountKey = accounts.currentAccountKey() ?: return FolderMutationOutcome.SignedOut
        val existing = folders.find(accountKey, id) ?: return FolderMutationOutcome.NotFound
        val normalized = name.trim()
        validateName(normalized)?.let { return it }
        val duplicate = folders.findByName(accountKey, normalized)
        if (duplicate != null && duplicate.id != id) return duplicateName()
        if (existing.name == normalized) return FolderMutationOutcome.Saved(existing)
        val updated = existing.copy(name = normalized, updatedAt = clock.now())
        folders.save(updated)
        return FolderMutationOutcome.Saved(updated)
    }

    override suspend fun deleteFolder(id: FolderId): FolderMutationOutcome {
        val accountKey = accounts.currentAccountKey() ?: return FolderMutationOutcome.SignedOut
        if (folders.find(accountKey, id) == null) return FolderMutationOutcome.NotFound
        val noteCount = notesRepository.countInFolder(accountKey, id)
        if (noteCount > 0) return FolderMutationOutcome.NotEmpty(noteCount)
        return if (folders.delete(accountKey, id)) FolderMutationOutcome.Deleted else FolderMutationOutcome.NotFound
    }

    override suspend fun moveNote(noteId: NoteId, folderId: FolderId?): MoveNoteOutcome {
        val accountKey = accounts.currentAccountKey() ?: return MoveNoteOutcome.SignedOut
        if (folderId != null && folders.find(accountKey, folderId) == null) return MoveNoteOutcome.FolderNotFound
        return when (val outcome = notesService.update(UpdateNoteCommand(noteId, folderId = FieldUpdate.Set(folderId)))) {
            is NoteMutationOutcome.Saved -> MoveNoteOutcome.Moved(outcome.note)
            NoteMutationOutcome.SignedOut -> MoveNoteOutcome.SignedOut
            NoteMutationOutcome.NotFound -> MoveNoteOutcome.NoteNotFound
            is NoteMutationOutcome.Invalid -> MoveNoteOutcome.Invalid(outcome.reason)
            NoteMutationOutcome.Deleted -> MoveNoteOutcome.Invalid("unexpected delete result")
        }
    }

    private fun validateName(name: String): FolderMutationOutcome.Invalid? = when {
        name.isBlank() -> FolderMutationOutcome.Invalid("name", "must not be blank")
        name.length > MAX_FOLDER_NAME_LENGTH ->
            FolderMutationOutcome.Invalid("name", "must not exceed $MAX_FOLDER_NAME_LENGTH characters")
        else -> null
    }

    private fun duplicateName() = FolderMutationOutcome.Invalid("name", "must be unique within account")

    private companion object {
        const val MAX_FOLDER_NAME_LENGTH = 120
    }
}

private fun List<AttachmentMetadata>.normalizeNoteId(noteId: NoteId): List<AttachmentMetadata> =
    map { it.copy(noteId = noteId) }

@Suppress("UNCHECKED_CAST")
private fun <T> FieldUpdate<T>.resolve(current: T): T = when (this) {
    FieldUpdate.Keep -> current
    is FieldUpdate.Set -> value
}

data class NotesCoreComponent(val notesService: NotesService, val folderService: FolderService) {
    companion object {
        fun create(
            accounts: AccountScope,
            notesRepository: NotesRepository,
            folderRepository: FolderRepository,
            clock: AppClock,
        ): NotesCoreComponent {
            val notes = DefaultNotesService(accounts, notesRepository, folderRepository, clock)
            val folders = DefaultFolderService(accounts, folderRepository, notesRepository, notes, clock)
            return NotesCoreComponent(notes, folders)
        }
    }
}

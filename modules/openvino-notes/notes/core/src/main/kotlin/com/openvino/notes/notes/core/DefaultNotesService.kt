// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package com.openvino.notes.notes.core

import com.openvino.notes.identity.api.AuthenticationState
import com.openvino.notes.identity.api.SessionReader
import com.openvino.notes.kernel.AppClock
import com.openvino.notes.notes.api.AttachmentMetadata
import com.openvino.notes.notes.api.ContentItem
import com.openvino.notes.notes.api.FieldUpdate
import com.openvino.notes.notes.api.Note
import com.openvino.notes.notes.api.NoteDraft
import com.openvino.notes.notes.api.NoteId
import com.openvino.notes.notes.api.NoteMutationOutcome
import com.openvino.notes.notes.api.NotesRepository
import com.openvino.notes.notes.api.NotesService
import com.openvino.notes.notes.api.UpdateNoteCommand
import java.util.UUID
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flatMapLatest
import kotlinx.coroutines.flow.flowOf

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
            is AuthenticationState.SignedIn -> repository.observe(state.session.accountKey)
        }
    }

    override suspend fun get(id: NoteId): Note? =
        sessions.currentAccountKey()?.let { repository.find(it, id) }

    override suspend fun create(draft: NoteDraft): NoteMutationOutcome {
        val accountKey = sessions.currentAccountKey() ?: return NoteMutationOutcome.SignedOut
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
        repository.save(note)
        return NoteMutationOutcome.Saved(note)
    }

    override suspend fun update(command: UpdateNoteCommand): NoteMutationOutcome {
        val accountKey = sessions.currentAccountKey() ?: return NoteMutationOutcome.SignedOut
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
        if (candidate == existing) return NoteMutationOutcome.Saved(existing)

        val updated = candidate.copy(updatedAt = clock.now())
        repository.save(updated)
        return NoteMutationOutcome.Saved(updated)
    }

    override suspend fun delete(id: NoteId): NoteMutationOutcome {
        val accountKey = sessions.currentAccountKey() ?: return NoteMutationOutcome.SignedOut
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

    private companion object {
        const val MAX_TITLE_LENGTH = 200
        const val MAX_TEXT_LENGTH = 100_000
    }
}

private fun List<AttachmentMetadata>.normalizeNoteId(noteId: NoteId): List<AttachmentMetadata> =
    map { it.copy(noteId = noteId) }

@Suppress("UNCHECKED_CAST")
private fun <T> FieldUpdate<T>.resolve(current: T): T = when (this) {
    FieldUpdate.Keep -> current
    is FieldUpdate.Set -> value
}

data class NotesCoreComponent(val notesService: NotesService) {
    companion object {
        fun create(sessions: SessionReader, repository: NotesRepository, clock: AppClock): NotesCoreComponent =
            NotesCoreComponent(DefaultNotesService(sessions, repository, clock))
    }
}

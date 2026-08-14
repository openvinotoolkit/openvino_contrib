// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package com.openvino.notes.notes.api

import com.openvino.notes.kernel.AccountKey
import java.time.Instant
import kotlinx.coroutines.flow.Flow

@JvmInline value class NoteId(val value: String)
@JvmInline value class FolderId(val value: String)
@JvmInline value class ContentItemId(val value: String)
@JvmInline value class AttachmentId(val value: String)
@JvmInline value class RemoteRevision(val value: String)

@JvmInline
value class NoteTag(val value: String) {
    init { require(value.isNotBlank()) { "NoteTag must not be blank" } }
}

sealed interface ContentItem {
    val id: ContentItemId

    data class Text(override val id: ContentItemId, val text: String) : ContentItem
    data class Image(override val id: ContentItemId, val attachmentId: AttachmentId, val caption: String? = null) : ContentItem
    data class File(override val id: ContentItemId, val attachmentId: AttachmentId) : ContentItem
    data class Link(override val id: ContentItemId, val url: String, val label: String? = null) : ContentItem
}

data class AttachmentMetadata(
    val id: AttachmentId,
    val noteId: NoteId,
    val contentItemId: ContentItemId,
    val displayName: String,
    val mediaType: String,
    val sizeBytes: Long,
)

data class Note(
    val id: NoteId,
    val accountKey: AccountKey,
    val title: String,
    val contentItems: List<ContentItem>,
    val attachments: List<AttachmentMetadata> = emptyList(),
    val folderId: FolderId? = null,
    val tags: Set<NoteTag> = emptySet(),
    val isFavorite: Boolean = false,
    val summary: String? = null,
    val createdAt: Instant,
    val updatedAt: Instant,
)

data class NoteDraft(
    val title: String,
    val contentItems: List<ContentItem>,
    val attachments: List<AttachmentMetadata> = emptyList(),
    val folderId: FolderId? = null,
    val tags: Set<NoteTag> = emptySet(),
    val isFavorite: Boolean = false,
    val summary: String? = null,
)

sealed interface FieldUpdate<out T> {
    data object Keep : FieldUpdate<Nothing>
    data class Set<T>(val value: T) : FieldUpdate<T>
}

data class UpdateNoteCommand(
    val id: NoteId,
    val title: FieldUpdate<String> = FieldUpdate.Keep,
    val contentItems: FieldUpdate<List<ContentItem>> = FieldUpdate.Keep,
    val attachments: FieldUpdate<List<AttachmentMetadata>> = FieldUpdate.Keep,
    val folderId: FieldUpdate<FolderId?> = FieldUpdate.Keep,
    val tags: FieldUpdate<Set<NoteTag>> = FieldUpdate.Keep,
    val isFavorite: FieldUpdate<Boolean> = FieldUpdate.Keep,
    val summary: FieldUpdate<String?> = FieldUpdate.Keep,
)

sealed interface NoteMutationOutcome {
    data class Saved(val note: Note) : NoteMutationOutcome
    data object Deleted : NoteMutationOutcome
    data object SignedOut : NoteMutationOutcome
    data object NotFound : NoteMutationOutcome
    data class Invalid(val field: String, val reason: String) : NoteMutationOutcome
}

interface NotesService {
    fun observeNotes(): Flow<List<Note>>
    suspend fun get(id: NoteId): Note?
    suspend fun create(draft: NoteDraft): NoteMutationOutcome
    suspend fun update(command: UpdateNoteCommand): NoteMutationOutcome
    suspend fun delete(id: NoteId): NoteMutationOutcome
}

interface NotesRepository {
    fun observe(accountKey: AccountKey): Flow<List<Note>>
    suspend fun find(accountKey: AccountKey, id: NoteId): Note?
    suspend fun save(note: Note)
    suspend fun delete(accountKey: AccountKey, id: NoteId): Boolean
}

enum class LocalChangeKind { UPSERT, DELETE }

data class LocalNoteChange(
    val changeId: String,
    val accountKey: AccountKey,
    val noteId: NoteId,
    val kind: LocalChangeKind,
    val baseRevision: RemoteRevision?,
    val changedAt: Instant,
    val payload: Note? = null,
)

sealed interface RemoteNoteChange {
    val noteId: NoteId?

    data class Upsert(
        val note: Note,
        val baseRevision: RemoteRevision?,
        val revision: RemoteRevision,
    ) : RemoteNoteChange {
        override val noteId: NoteId = note.id
    }

    data class Tombstone(
        override val noteId: NoteId,
        val revision: RemoteRevision,
    ) : RemoteNoteChange

    data class Malformed(
        override val noteId: NoteId?,
        val diagnosticCode: String,
    ) : RemoteNoteChange
}

sealed interface RemoteApplyResult {
    val noteId: NoteId?
    data class Applied(override val noteId: NoteId, val revision: RemoteRevision) : RemoteApplyResult
    data class Merged(override val noteId: NoteId, val generatedChangeId: String) : RemoteApplyResult
    data class Conflict(override val noteId: NoteId, val localRevision: RemoteRevision?, val remoteRevision: RemoteRevision) : RemoteApplyResult
    data class TombstoneApplied(override val noteId: NoteId, val revision: RemoteRevision) : RemoteApplyResult
    data class RejectedMalformed(override val noteId: NoteId?, val diagnosticCode: String) : RemoteApplyResult
}

interface NotesSyncPort {
    suspend fun pendingChanges(accountKey: AccountKey, limit: Int = 100): List<LocalNoteChange>
    suspend fun acknowledge(accountKey: AccountKey, changeIds: Set<String>)
    suspend fun applyRemote(accountKey: AccountKey, changes: List<RemoteNoteChange>): List<RemoteApplyResult>
}

fun interface BinarySource {
    suspend fun read(): ByteArray
}

interface AttachmentContentPort {
    suspend fun put(accountKey: AccountKey, attachment: AttachmentMetadata, source: BinarySource)
    suspend fun open(accountKey: AccountKey, attachmentId: AttachmentId): BinarySource?
}

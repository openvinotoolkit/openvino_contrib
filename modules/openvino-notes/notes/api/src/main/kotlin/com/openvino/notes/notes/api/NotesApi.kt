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

data class Folder(
    val id: FolderId,
    val accountKey: AccountKey,
    val name: String,
    val createdAt: Instant,
    val updatedAt: Instant,
)

data class FolderDraft(val name: String)

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

sealed interface FolderMutationOutcome {
    data class Saved(val folder: Folder) : FolderMutationOutcome
    data object Deleted : FolderMutationOutcome
    data object SignedOut : FolderMutationOutcome
    data object NotFound : FolderMutationOutcome
    data class NotEmpty(val noteCount: Int) : FolderMutationOutcome
    data class Invalid(val field: String, val reason: String) : FolderMutationOutcome
}

sealed interface MoveNoteOutcome {
    data class Moved(val note: Note) : MoveNoteOutcome
    data object SignedOut : MoveNoteOutcome
    data object NoteNotFound : MoveNoteOutcome
    data object FolderNotFound : MoveNoteOutcome
    data class Invalid(val reason: String) : MoveNoteOutcome
}

interface FolderService {
    fun observeFolders(): Flow<List<Folder>>
    suspend fun getFolder(id: FolderId): Folder?
    suspend fun createFolder(draft: FolderDraft): FolderMutationOutcome
    suspend fun renameFolder(id: FolderId, name: String): FolderMutationOutcome
    suspend fun deleteFolder(id: FolderId): FolderMutationOutcome
    suspend fun moveNote(noteId: NoteId, folderId: FolderId?): MoveNoteOutcome
}

package com.itlab.notes.api

import com.itlab.identity.api.AccountId
import java.time.Instant
import kotlinx.coroutines.flow.Flow

@JvmInline value class NoteId(val value: String)
@JvmInline value class FolderId(val value: String)
@JvmInline value class AttachmentId(val value: String)

data class Note(
    val id: NoteId,
    val accountId: AccountId,
    val title: String,
    val body: String,
    val folderId: FolderId? = null,
    val updatedAt: Instant,
)

data class Folder(val id: FolderId, val accountId: AccountId, val name: String)

data class Attachment(
    val id: AttachmentId,
    val noteId: NoteId,
    val displayName: String,
    val mediaType: String,
    val sizeBytes: Long,
)

data class NoteDraft(val title: String, val body: String, val folderId: FolderId? = null)
data class UpdateNoteCommand(val id: NoteId, val title: String, val body: String, val folderId: FolderId? = null)

sealed interface NoteMutationOutcome {
    data class Saved(val note: Note) : NoteMutationOutcome
    data object Deleted : NoteMutationOutcome
    data object SignedOut : NoteMutationOutcome
    data object NotFound : NoteMutationOutcome
    data class Invalid(val field: String, val reason: String) : NoteMutationOutcome
}

interface NotesService {
    fun observeNotes(): Flow<List<Note>>
    suspend fun create(draft: NoteDraft): NoteMutationOutcome
    suspend fun update(command: UpdateNoteCommand): NoteMutationOutcome
    suspend fun delete(id: NoteId): NoteMutationOutcome
}

interface NotesRepository {
    fun observe(accountId: AccountId): Flow<List<Note>>
    suspend fun find(accountId: AccountId, id: NoteId): Note?
    suspend fun save(note: Note)
    suspend fun delete(accountId: AccountId, id: NoteId): Boolean
}

enum class LocalChangeKind { UPSERT, DELETE }

data class LocalNoteChange(
    val changeId: String,
    val accountId: AccountId,
    val noteId: NoteId,
    val kind: LocalChangeKind,
    val changedAt: Instant,
    val payload: Note? = null,
)

data class RemoteNoteChange(
    val remoteId: String,
    val noteId: NoteId,
    val deleted: Boolean,
    val updatedAt: Instant,
    val note: Note? = null,
)

interface NotesSyncPort {
    suspend fun pendingChanges(accountId: AccountId, limit: Int = 100): List<LocalNoteChange>
    suspend fun acknowledge(changeIds: Set<String>)
    suspend fun applyRemote(accountId: AccountId, changes: List<RemoteNoteChange>)
}

fun interface BinarySource {
    suspend fun read(): ByteArray
}

interface AttachmentContentPort {
    suspend fun put(accountId: AccountId, attachment: Attachment, source: BinarySource)
    suspend fun open(accountId: AccountId, attachmentId: AttachmentId): BinarySource?
}

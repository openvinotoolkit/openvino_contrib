// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package com.openvino.notes.notes.room

import android.content.Context
import androidx.room.Dao
import androidx.room.Database
import androidx.room.Entity
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import androidx.room.Room
import androidx.room.RoomDatabase
import androidx.room.Transaction
import com.openvino.notes.kernel.AccountKey
import com.openvino.notes.notes.api.AttachmentId
import com.openvino.notes.notes.api.AttachmentMetadata
import com.openvino.notes.notes.api.ContentItem
import com.openvino.notes.notes.api.ContentItemId
import com.openvino.notes.notes.api.FolderId
import com.openvino.notes.notes.api.LocalChangeKind
import com.openvino.notes.notes.api.LocalNoteChange
import com.openvino.notes.notes.api.Note
import com.openvino.notes.notes.api.NoteId
import com.openvino.notes.notes.api.NoteTag
import com.openvino.notes.notes.api.NotesRepository
import com.openvino.notes.notes.api.NotesSyncPort
import com.openvino.notes.notes.api.RemoteApplyResult
import com.openvino.notes.notes.api.RemoteNoteChange
import com.openvino.notes.notes.api.RemoteRevision
import java.nio.charset.StandardCharsets
import java.time.Instant
import java.util.Base64
import java.util.UUID
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map

@Entity(tableName = "notes", primaryKeys = ["accountKey", "id"])
internal data class NoteEntity(
    val accountKey: String,
    val id: String,
    val title: String,
    val contentItems: String,
    val attachments: String,
    val folderId: String?,
    val tags: String,
    val isFavorite: Boolean,
    val summary: String?,
    val createdAtMillis: Long,
    val updatedAtMillis: Long,
)

@Entity(tableName = "note_outbox", primaryKeys = ["accountKey", "changeId"])
internal data class OutboxEntity(
    val accountKey: String,
    val changeId: String,
    val noteId: String,
    val kind: String,
    val baseRevision: String?,
    val changedAtMillis: Long,
    val title: String?,
    val contentItems: String?,
    val attachments: String?,
    val folderId: String?,
    val tags: String?,
    val isFavorite: Boolean?,
    val summary: String?,
    val createdAtMillis: Long?,
)

@Entity(tableName = "note_remote_revision", primaryKeys = ["accountKey", "noteId"])
internal data class RemoteRevisionEntity(
    val accountKey: String,
    val noteId: String,
    val revision: String,
)

@Dao
internal interface NotesDao {
    @Query("SELECT * FROM notes WHERE accountKey = :accountKey ORDER BY updatedAtMillis DESC")
    fun observe(accountKey: String): Flow<List<NoteEntity>>

    @Query("SELECT * FROM notes WHERE accountKey = :accountKey AND id = :id")
    suspend fun find(accountKey: String, id: String): NoteEntity?

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun upsert(entity: NoteEntity)

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun addOutbox(entity: OutboxEntity)

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun saveRevision(entity: RemoteRevisionEntity)

    @Query("SELECT * FROM note_remote_revision WHERE accountKey = :accountKey AND noteId = :noteId")
    suspend fun revision(accountKey: String, noteId: String): RemoteRevisionEntity?

    @Query("DELETE FROM notes WHERE accountKey = :accountKey AND id = :id")
    suspend fun deleteNote(accountKey: String, id: String): Int

    @Query("SELECT COUNT(*) FROM note_outbox WHERE accountKey = :accountKey AND noteId = :noteId")
    suspend fun pendingCount(accountKey: String, noteId: String): Int

    @Query("SELECT * FROM note_outbox WHERE accountKey = :accountKey ORDER BY changedAtMillis LIMIT :limit")
    suspend fun pending(accountKey: String, limit: Int): List<OutboxEntity>

    @Query("DELETE FROM note_outbox WHERE accountKey = :accountKey AND changeId IN (:changeIds)")
    suspend fun acknowledge(accountKey: String, changeIds: Set<String>)

    @Transaction
    suspend fun saveLocally(note: NoteEntity, outbox: OutboxEntity) {
        upsert(note)
        addOutbox(outbox.copy(baseRevision = revision(note.accountKey, note.id)?.revision))
    }

    @Transaction
    suspend fun deleteLocally(accountKey: String, id: String, outbox: OutboxEntity): Boolean {
        val removed = deleteNote(accountKey, id) > 0
        if (removed) addOutbox(outbox.copy(baseRevision = revision(accountKey, id)?.revision))
        return removed
    }

    @Transaction
    suspend fun applyRemoteUpsert(note: NoteEntity, remoteRevision: String): Boolean {
        if (pendingCount(note.accountKey, note.id) > 0) return false
        upsert(note)
        saveRevision(RemoteRevisionEntity(note.accountKey, note.id, remoteRevision))
        return true
    }

    @Transaction
    suspend fun applyRemoteTombstone(accountKey: String, noteId: String, remoteRevision: String): Boolean {
        if (pendingCount(accountKey, noteId) > 0) return false
        deleteNote(accountKey, noteId)
        saveRevision(RemoteRevisionEntity(accountKey, noteId, remoteRevision))
        return true
    }
}

@Database(
    entities = [NoteEntity::class, OutboxEntity::class, RemoteRevisionEntity::class],
    version = 1,
    exportSchema = true,
)
internal abstract class NotesDatabase : RoomDatabase() {
    abstract fun notesDao(): NotesDao
}

internal class RoomNotesRepository(private val dao: NotesDao) : NotesRepository, NotesSyncPort {
    override fun observe(accountKey: AccountKey): Flow<List<Note>> =
        dao.observe(accountKey.value).map { entities -> entities.map(NoteEntity::toApi) }

    override suspend fun find(accountKey: AccountKey, id: NoteId): Note? =
        dao.find(accountKey.value, id.value)?.toApi()

    override suspend fun save(note: Note) {
        dao.saveLocally(note.toEntity(), note.toOutbox(LocalChangeKind.UPSERT))
    }

    override suspend fun delete(accountKey: AccountKey, id: NoteId): Boolean = dao.deleteLocally(
        accountKey.value,
        id.value,
        OutboxEntity(
            accountKey = accountKey.value,
            changeId = UUID.randomUUID().toString(),
            noteId = id.value,
            kind = LocalChangeKind.DELETE.name,
            baseRevision = null,
            changedAtMillis = System.currentTimeMillis(),
            title = null,
            contentItems = null,
            attachments = null,
            folderId = null,
            tags = null,
            isFavorite = null,
            summary = null,
            createdAtMillis = null,
        ),
    )

    override suspend fun pendingChanges(accountKey: AccountKey, limit: Int): List<LocalNoteChange> =
        dao.pending(accountKey.value, limit).map(OutboxEntity::toApi)

    override suspend fun acknowledge(accountKey: AccountKey, changeIds: Set<String>) {
        if (changeIds.isNotEmpty()) dao.acknowledge(accountKey.value, changeIds)
    }

    override suspend fun applyRemote(
        accountKey: AccountKey,
        changes: List<RemoteNoteChange>,
    ): List<RemoteApplyResult> = changes.map { change ->
        when (change) {
            is RemoteNoteChange.Malformed -> RemoteApplyResult.RejectedMalformed(
                change.noteId,
                change.diagnosticCode,
            )
            is RemoteNoteChange.Upsert -> applyUpsert(accountKey, change)
            is RemoteNoteChange.Tombstone -> applyTombstone(accountKey, change)
        }
    }

    private suspend fun applyUpsert(accountKey: AccountKey, change: RemoteNoteChange.Upsert): RemoteApplyResult {
        if (change.note.accountKey != accountKey) {
            return RemoteApplyResult.RejectedMalformed(change.note.id, "notes.remote.account_mismatch")
        }
        return if (dao.applyRemoteUpsert(change.note.toEntity(), change.revision.value)) {
            RemoteApplyResult.Applied(change.note.id, change.revision)
        } else {
            RemoteApplyResult.Conflict(change.note.id, localRevision(accountKey, change.note.id), change.revision)
        }
    }

    private suspend fun applyTombstone(
        accountKey: AccountKey,
        change: RemoteNoteChange.Tombstone,
    ): RemoteApplyResult = if (dao.applyRemoteTombstone(accountKey.value, change.noteId.value, change.revision.value)) {
        RemoteApplyResult.TombstoneApplied(change.noteId, change.revision)
    } else {
        RemoteApplyResult.Conflict(change.noteId, localRevision(accountKey, change.noteId), change.revision)
    }

    private suspend fun localRevision(accountKey: AccountKey, noteId: NoteId): RemoteRevision? =
        dao.revision(accountKey.value, noteId.value)?.revision?.let(::RemoteRevision)
}

class RoomNotesComponent private constructor(
    private val database: NotesDatabase,
    val repository: NotesRepository,
    val syncPort: NotesSyncPort,
) : AutoCloseable {
    override fun close() = database.close()

    companion object {
        fun create(context: Context, databaseName: String = "openvino-notes.db"): RoomNotesComponent {
            val database = Room.databaseBuilder(context.applicationContext, NotesDatabase::class.java, databaseName).build()
            val implementation = RoomNotesRepository(database.notesDao())
            return RoomNotesComponent(database, implementation, implementation)
        }
    }
}

private fun Note.toEntity() = NoteEntity(
    accountKey = accountKey.value,
    id = id.value,
    title = title,
    contentItems = WireCodec.encodeContent(contentItems),
    attachments = WireCodec.encodeAttachments(attachments),
    folderId = folderId?.value,
    tags = WireCodec.encodeTags(tags),
    isFavorite = isFavorite,
    summary = summary,
    createdAtMillis = createdAt.toEpochMilli(),
    updatedAtMillis = updatedAt.toEpochMilli(),
)

private fun NoteEntity.toApi() = Note(
    id = NoteId(id),
    accountKey = AccountKey(accountKey),
    title = title,
    contentItems = WireCodec.decodeContent(contentItems),
    attachments = WireCodec.decodeAttachments(attachments),
    folderId = folderId?.let(::FolderId),
    tags = WireCodec.decodeTags(tags),
    isFavorite = isFavorite,
    summary = summary,
    createdAt = Instant.ofEpochMilli(createdAtMillis),
    updatedAt = Instant.ofEpochMilli(updatedAtMillis),
)

private fun Note.toOutbox(kind: LocalChangeKind) = OutboxEntity(
    accountKey = accountKey.value,
    changeId = UUID.randomUUID().toString(),
    noteId = id.value,
    kind = kind.name,
    baseRevision = null,
    changedAtMillis = updatedAt.toEpochMilli(),
    title = title,
    contentItems = WireCodec.encodeContent(contentItems),
    attachments = WireCodec.encodeAttachments(attachments),
    folderId = folderId?.value,
    tags = WireCodec.encodeTags(tags),
    isFavorite = isFavorite,
    summary = summary,
    createdAtMillis = createdAt.toEpochMilli(),
)

private fun OutboxEntity.toApi(): LocalNoteChange {
    val changeKind = LocalChangeKind.valueOf(kind)
    val note = if (changeKind == LocalChangeKind.UPSERT) {
        Note(
            id = NoteId(noteId),
            accountKey = AccountKey(accountKey),
            title = requireNotNull(title),
            contentItems = WireCodec.decodeContent(requireNotNull(contentItems)),
            attachments = WireCodec.decodeAttachments(requireNotNull(attachments)),
            folderId = folderId?.let(::FolderId),
            tags = WireCodec.decodeTags(requireNotNull(tags)),
            isFavorite = requireNotNull(isFavorite),
            summary = summary,
            createdAt = Instant.ofEpochMilli(requireNotNull(createdAtMillis)),
            updatedAt = Instant.ofEpochMilli(changedAtMillis),
        )
    } else {
        null
    }
    return LocalNoteChange(
        changeId = changeId,
        accountKey = AccountKey(accountKey),
        noteId = NoteId(noteId),
        kind = changeKind,
        baseRevision = baseRevision?.let(::RemoteRevision),
        changedAt = Instant.ofEpochMilli(changedAtMillis),
        payload = note,
    )
}

private object WireCodec {
    private val encoder = Base64.getUrlEncoder().withoutPadding()
    private val decoder = Base64.getUrlDecoder()

    fun encodeContent(items: List<ContentItem>): String = items.joinToString("\n") { item ->
        when (item) {
            is ContentItem.Text -> listOf("text", item.id.value, encode(item.text)).joinToString("|")
            is ContentItem.Image -> listOf("image", item.id.value, item.attachmentId.value, encodeNullable(item.caption)).joinToString("|")
            is ContentItem.File -> listOf("file", item.id.value, item.attachmentId.value).joinToString("|")
            is ContentItem.Link -> listOf("link", item.id.value, encode(item.url), encodeNullable(item.label)).joinToString("|")
        }
    }

    fun decodeContent(value: String): List<ContentItem> = lines(value).map { line ->
        val fields = line.split('|')
        when (fields.firstOrNull()) {
            "text" -> ContentItem.Text(ContentItemId(fields[1]), decode(fields[2]))
            "image" -> ContentItem.Image(ContentItemId(fields[1]), AttachmentId(fields[2]), decodeNullable(fields[3]))
            "file" -> ContentItem.File(ContentItemId(fields[1]), AttachmentId(fields[2]))
            "link" -> ContentItem.Link(ContentItemId(fields[1]), decode(fields[2]), decodeNullable(fields[3]))
            else -> error("Unsupported content item")
        }
    }

    fun encodeAttachments(items: List<AttachmentMetadata>): String = items.joinToString("\n") { item ->
        listOf(
            item.id.value,
            item.noteId.value,
            item.contentItemId.value,
            encode(item.displayName),
            encode(item.mediaType),
            item.sizeBytes.toString(),
        ).joinToString("|")
    }

    fun decodeAttachments(value: String): List<AttachmentMetadata> = lines(value).map { line ->
        val fields = line.split('|')
        AttachmentMetadata(
            id = AttachmentId(fields[0]),
            noteId = NoteId(fields[1]),
            contentItemId = ContentItemId(fields[2]),
            displayName = decode(fields[3]),
            mediaType = decode(fields[4]),
            sizeBytes = fields[5].toLong(),
        )
    }

    fun encodeTags(tags: Set<NoteTag>): String = tags.map(NoteTag::value).sorted().joinToString("\n", transform = ::encode)
    fun decodeTags(value: String): Set<NoteTag> = lines(value).map { NoteTag(decode(it)) }.toSet()

    private fun lines(value: String): List<String> = if (value.isEmpty()) emptyList() else value.split('\n')
    private fun encode(value: String): String = encoder.encodeToString(value.toByteArray(StandardCharsets.UTF_8))
    private fun decode(value: String): String = String(decoder.decode(value), StandardCharsets.UTF_8)
    private fun encodeNullable(value: String?): String = value?.let { "+${encode(it)}" } ?: "-"
    private fun decodeNullable(value: String): String? = if (value == "-") null else decode(value.removePrefix("+"))
}

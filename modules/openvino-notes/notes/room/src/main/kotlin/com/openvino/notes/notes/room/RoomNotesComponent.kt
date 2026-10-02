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
import com.openvino.notes.kernel.AppDispatchers
import com.openvino.notes.notes.api.AttachmentId
import com.openvino.notes.notes.api.AttachmentMetadata
import com.openvino.notes.notes.api.ContentItem
import com.openvino.notes.notes.api.ContentItemId
import com.openvino.notes.notes.api.FolderId
import com.openvino.notes.notes.api.Note
import com.openvino.notes.notes.api.NoteId
import com.openvino.notes.notes.api.NoteTag
import com.openvino.notes.notes.api.port.AttachmentContentPort
import com.openvino.notes.notes.api.port.LocalChangeKind
import com.openvino.notes.notes.api.port.LocalNoteChange
import com.openvino.notes.notes.api.port.FolderRepository
import com.openvino.notes.notes.api.port.FolderSyncPort
import com.openvino.notes.notes.api.port.NotesRepository
import com.openvino.notes.notes.api.port.NotesSyncPort
import com.openvino.notes.notes.api.port.RemoteApplyResult
import com.openvino.notes.notes.api.port.RemoteNoteChange
import com.openvino.notes.notes.api.port.RemoteRevision
import java.io.File
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

    @Query("SELECT COUNT(*) FROM notes WHERE accountKey = :accountKey AND folderId = :folderId")
    suspend fun countInFolder(accountKey: String, folderId: String): Int

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
    entities = [
        NoteEntity::class,
        OutboxEntity::class,
        RemoteRevisionEntity::class,
        FolderEntity::class,
        FolderOutboxEntity::class,
        FolderRemoteRevisionEntity::class,
    ],
    version = 1,
    exportSchema = true,
)
internal abstract class NotesDatabase : RoomDatabase() {
    abstract fun notesDao(): NotesDao
    abstract fun folderDao(): FolderDao
}

internal class RoomNotesRepository(
    private val dao: NotesDao,
    private val attachmentContent: AttachmentContentPort,
) : NotesRepository, NotesSyncPort {
    override fun observe(accountKey: AccountKey): Flow<List<Note>> =
        dao.observe(accountKey.value).map { entities -> entities.map(NoteEntity::toApi) }

    override suspend fun find(accountKey: AccountKey, id: NoteId): Note? =
        dao.find(accountKey.value, id.value)?.toApi()

    override suspend fun countInFolder(accountKey: AccountKey, folderId: FolderId): Int =
        dao.countInFolder(accountKey.value, folderId.value)

    override suspend fun save(note: Note) {
        val previous = dao.find(note.accountKey.value, note.id.value)?.toApi()
        dao.saveLocally(note.toEntity(), note.toOutbox(LocalChangeKind.UPSERT))
        previous?.attachments
            ?.map(AttachmentMetadata::id)
            ?.filter { previousId -> note.attachments.none { it.id == previousId } }
            ?.forEach { attachmentContent.delete(note.accountKey, it) }
    }

    override suspend fun delete(accountKey: AccountKey, id: NoteId): Boolean {
        val previous = dao.find(accountKey.value, id.value)?.toApi()
        val removed = dao.deleteLocally(
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
        if (removed) previous?.attachments?.forEach { attachmentContent.delete(accountKey, it.id) }
        return removed
    }

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
        val previous = dao.find(accountKey.value, change.note.id.value)?.toApi()
        return if (dao.applyRemoteUpsert(change.note.toEntity(), change.revision.value)) {
            previous?.attachments
                ?.map(AttachmentMetadata::id)
                ?.filter { previousId -> change.note.attachments.none { it.id == previousId } }
                ?.forEach { attachmentContent.delete(accountKey, it) }
            RemoteApplyResult.Applied(change.note.id, change.revision)
        } else {
            RemoteApplyResult.Conflict(change.note.id, localRevision(accountKey, change.note.id), change.revision)
        }
    }

    private suspend fun applyTombstone(
        accountKey: AccountKey,
        change: RemoteNoteChange.Tombstone,
    ): RemoteApplyResult {
        val previous = dao.find(accountKey.value, change.noteId.value)?.toApi()
        return if (dao.applyRemoteTombstone(accountKey.value, change.noteId.value, change.revision.value)) {
            previous?.attachments?.forEach { attachmentContent.delete(accountKey, it.id) }
            RemoteApplyResult.TombstoneApplied(change.noteId, change.revision)
        } else {
            RemoteApplyResult.Conflict(change.noteId, localRevision(accountKey, change.noteId), change.revision)
        }
    }

    private suspend fun localRevision(accountKey: AccountKey, noteId: NoteId): RemoteRevision? =
        dao.revision(accountKey.value, noteId.value)?.revision?.let(::RemoteRevision)
}

class RoomNotesComponent private constructor(
    private val database: NotesDatabase,
    val repository: NotesRepository,
    val syncPort: NotesSyncPort,
    val folderRepository: FolderRepository,
    val folderSyncPort: FolderSyncPort,
    val attachmentContent: AttachmentContentPort,
) : AutoCloseable {
    override fun close() = database.close()

    companion object {
        fun create(
            context: Context,
            databaseName: String = "openvino-notes.db",
            dispatchers: AppDispatchers = AppDispatchers.production(),
        ): RoomNotesComponent {
            val database = Room.databaseBuilder(context.applicationContext, NotesDatabase::class.java, databaseName).build()
            val attachmentContent = FileAttachmentContentStore(
                File(context.applicationContext.filesDir, "notes-media"),
                dispatchers,
            )
            val notes = RoomNotesRepository(database.notesDao(), attachmentContent)
            val folders = RoomFolderRepository(database.folderDao())
            return RoomNotesComponent(database, notes, notes, folders, folders, attachmentContent)
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
    private const val VERSION_2_MARKER = "v2:"
    private val encoder = Base64.getUrlEncoder().withoutPadding()
    private val decoder = Base64.getUrlDecoder()

    fun encodeContent(items: List<ContentItem>): String {
        if (items.isEmpty()) return VERSION_2_MARKER
        return buildString {
            append(VERSION_2_MARKER)
            items.forEach { item ->
                append('\n')
                when (item) {
                    is ContentItem.Text -> listOf(
                        "text",
                        encodeString(item.id.value),
                        encodeString(item.text),
                    )

                    is ContentItem.Image -> listOf(
                        "image",
                        encodeString(item.id.value),
                        encodeString(item.attachmentId.value),
                        encodeNullableString(item.caption),
                    )

                    is ContentItem.File -> listOf(
                        "file",
                        encodeString(item.id.value),
                        encodeString(item.attachmentId.value),
                    )

                    is ContentItem.Link -> listOf(
                        "link",
                        encodeString(item.id.value),
                        encodeString(item.url),
                        encodeNullableString(item.label),
                    )
                }.joinTo(this, "|")
            }
        }
    }

    fun decodeContent(value: String): List<ContentItem> =
        if (isVersion2(value)) decodeContentV2(value) else decodeContentV1(value)

    private fun decodeContentV1(value: String): List<ContentItem> =
        lines(value).map { line ->
            val fields = line.split('|')
            when (fields.firstOrNull()) {
                "text" -> ContentItem.Text(
                    ContentItemId(fields[1]),
                    decode(fields[2]),
                )

                "image" -> ContentItem.Image(
                    ContentItemId(fields[1]),
                    AttachmentId(fields[2]),
                    decodeNullable(fields[3]),
                )

                "file" -> ContentItem.File(
                    ContentItemId(fields[1]),
                    AttachmentId(fields[2]),
                )

                "link" -> ContentItem.Link(
                    ContentItemId(fields[1]),
                    decode(fields[2]),
                    decodeNullable(fields[3]),
                )

                else -> error("Unsupported content item")
            }
        }

    private fun decodeContentV2(value: String): List<ContentItem> =
        linesAfterMarker(value).map { line ->
            val fields = splitFields(line)

            when (fields.firstOrNull()) {
                "text" -> {
                    requireFieldCount(fields, 3)
                    ContentItem.Text(
                        id = ContentItemId(decodeString(fields[1])),
                        text = decodeString(fields[2]),
                    )
                }

                "image" -> {
                    requireFieldCount(fields, 4)
                    ContentItem.Image(
                        id = ContentItemId(decodeString(fields[1])),
                        attachmentId = AttachmentId(decodeString(fields[2])),
                        caption = decodeNullableString(fields[3]),
                    )
                }

                "file" -> {
                    requireFieldCount(fields, 3)
                    ContentItem.File(
                        id = ContentItemId(decodeString(fields[1])),
                        attachmentId = AttachmentId(decodeString(fields[2])),
                    )
                }

                "link" -> {
                    requireFieldCount(fields, 4)
                    ContentItem.Link(
                        id = ContentItemId(decodeString(fields[1])),
                        url = decodeString(fields[2]),
                        label = decodeNullableString(fields[3]),
                    )
                }

                else -> error("Unsupported content item")
            }
        }

    fun encodeAttachments(items: List<AttachmentMetadata>): String {
        if (items.isEmpty()) return VERSION_2_MARKER
        return buildString {
            append(VERSION_2_MARKER)
            items.forEach { item ->
                append('\n')
                listOf(
                    encodeString(item.id.value),
                    encodeString(item.noteId.value),
                    encodeString(item.contentItemId.value),
                    encodeString(item.displayName),
                    encodeString(item.mediaType),
                    item.sizeBytes.toString(),
                ).joinTo(this, "|")
            }
        }
    }

    fun decodeAttachments(value: String): List<AttachmentMetadata> =
        if (isVersion2(value)) decodeAttachmentsV2(value) else decodeAttachmentsV1(value)

    private fun decodeAttachmentsV1(value: String): List<AttachmentMetadata> =
        lines(value).map { line ->
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

    private fun decodeAttachmentsV2(value: String): List<AttachmentMetadata> =
        linesAfterMarker(value).map { line ->
            val fields = splitFields(line)
            requireFieldCount(fields, 6)
            AttachmentMetadata(
                id = AttachmentId(decodeString(fields[0])),
                noteId = NoteId(decodeString(fields[1])),
                contentItemId = ContentItemId(decodeString(fields[2])),
                displayName = decodeString(fields[3]),
                mediaType = decodeString(fields[4]),
                sizeBytes = decodeLong(fields[5]),
            )
        }

    fun encodeTags(tags: Set<NoteTag>): String {
        if (tags.isEmpty()) return VERSION_2_MARKER
        return buildString {
            append(VERSION_2_MARKER)
            tags.map(NoteTag::value)
                .sorted()
                .forEach {
                    append('\n')
                    append(encodeString(it))
                }
        }
    }

    fun decodeTags(value: String): Set<NoteTag> =
        if (isVersion2(value)) decodeTagsV2(value) else decodeTagsV1(value)

    private fun decodeTagsV1(value: String): Set<NoteTag> =
        lines(value).map { NoteTag(decode(it)) }.toSet()

    private fun decodeTagsV2(value: String): Set<NoteTag> =
        linesAfterMarker(value)
            .map { NoteTag(decodeString(it)) }
            .toSet()

    private fun isVersion2(value: String): Boolean =
        value == VERSION_2_MARKER || value.startsWith("$VERSION_2_MARKER\n")

    private fun lines(value: String): List<String> =
        if (value.isEmpty()) emptyList() else value.split('\n')

    private fun linesAfterMarker(value: String): List<String> =
        if (value == VERSION_2_MARKER) {
            emptyList()
        } else {
            value.removePrefix("$VERSION_2_MARKER\n").split('\n')
        }

    private fun splitFields(line: String): List<String> =
        line.split('|')

    private fun requireFieldCount(fields: List<String>, expected: Int) {
        require(fields.size == expected) {
            "Expected $expected fields, got ${fields.size}"
        }
    }

    private fun encodeString(value: String): String =
        encoder.encodeToString(value.toByteArray(StandardCharsets.UTF_8))

    private fun decodeString(value: String): String =
        String(decoder.decode(value), StandardCharsets.UTF_8)

    private fun encodeNullableString(value: String?): String =
        value?.let { "+${encodeString(it)}" } ?: "-"

    private fun decodeNullableString(value: String): String? =
        if (value == "-") null
        else {
            require(value.startsWith("+")) { "Invalid nullable string" }
            decodeString(value.removePrefix("+"))
        }

    private fun decodeLong(value: String): Long =
        value.toLong()

    private fun encode(value: String): String =
        encoder.encodeToString(value.toByteArray(StandardCharsets.UTF_8))

    private fun decode(value: String): String =
        String(decoder.decode(value), StandardCharsets.UTF_8)

    private fun decodeNullable(value: String): String? =
        if (value == "-") null else decode(value.removePrefix("+"))
}
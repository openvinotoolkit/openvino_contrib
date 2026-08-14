package com.itlab.notes.room

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
import com.itlab.identity.api.AccountId
import com.itlab.notes.api.LocalChangeKind
import com.itlab.notes.api.LocalNoteChange
import com.itlab.notes.api.FolderId
import com.itlab.notes.api.Note
import com.itlab.notes.api.NoteId
import com.itlab.notes.api.NotesRepository
import com.itlab.notes.api.NotesSyncPort
import com.itlab.notes.api.RemoteNoteChange
import java.time.Instant
import java.util.UUID
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map

@Entity(tableName = "notes", primaryKeys = ["accountId", "id"])
internal data class NoteEntity(
    val accountId: String,
    val id: String,
    val title: String,
    val body: String,
    val folderId: String?,
    val updatedAtMillis: Long,
)

@Entity(tableName = "note_outbox", primaryKeys = ["accountId", "changeId"])
internal data class OutboxEntity(
    val accountId: String,
    val changeId: String,
    val noteId: String,
    val kind: String,
    val changedAtMillis: Long,
    val title: String?,
    val body: String?,
    val folderId: String?,
)

@Dao
internal interface NotesDao {
    @Query("SELECT * FROM notes WHERE accountId = :accountId ORDER BY updatedAtMillis DESC")
    fun observe(accountId: String): Flow<List<NoteEntity>>

    @Query("SELECT * FROM notes WHERE accountId = :accountId AND id = :id")
    suspend fun find(accountId: String, id: String): NoteEntity?

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun upsert(entity: NoteEntity)

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun addOutbox(entity: OutboxEntity)

    @Query("DELETE FROM notes WHERE accountId = :accountId AND id = :id")
    suspend fun deleteNote(accountId: String, id: String): Int

    @Query("SELECT * FROM note_outbox WHERE accountId = :accountId ORDER BY changedAtMillis LIMIT :limit")
    suspend fun pending(accountId: String, limit: Int): List<OutboxEntity>

    @Query("DELETE FROM note_outbox WHERE changeId IN (:changeIds)")
    suspend fun acknowledge(changeIds: Set<String>)

    @Transaction
    suspend fun saveLocally(note: NoteEntity, outbox: OutboxEntity) {
        upsert(note)
        addOutbox(outbox)
    }

    @Transaction
    suspend fun deleteLocally(accountId: String, id: String, outbox: OutboxEntity): Boolean {
        val removed = deleteNote(accountId, id) > 0
        if (removed) addOutbox(outbox)
        return removed
    }
}

@Database(entities = [NoteEntity::class, OutboxEntity::class], version = 1, exportSchema = true)
internal abstract class NotesDatabase : RoomDatabase() {
    abstract fun notesDao(): NotesDao
}

internal class RoomNotesRepository(private val dao: NotesDao) : NotesRepository, NotesSyncPort {
    override fun observe(accountId: AccountId): Flow<List<Note>> = dao.observe(accountId.value).map { it.map(NoteEntity::toApi) }
    override suspend fun find(accountId: AccountId, id: NoteId): Note? = dao.find(accountId.value, id.value)?.toApi()
    override suspend fun save(note: Note) {
        dao.saveLocally(note.toEntity(), note.toOutbox(LocalChangeKind.UPSERT))
    }
    override suspend fun delete(accountId: AccountId, id: NoteId): Boolean = dao.deleteLocally(
        accountId.value,
        id.value,
        OutboxEntity(accountId.value, UUID.randomUUID().toString(), id.value, LocalChangeKind.DELETE.name, System.currentTimeMillis(), null, null, null),
    )
    override suspend fun pendingChanges(accountId: AccountId, limit: Int): List<LocalNoteChange> = dao.pending(accountId.value, limit).map(OutboxEntity::toApi)
    override suspend fun acknowledge(changeIds: Set<String>) { if (changeIds.isNotEmpty()) dao.acknowledge(changeIds) }
    override suspend fun applyRemote(accountId: AccountId, changes: List<RemoteNoteChange>) {
        changes.forEach { change ->
            if (change.deleted) dao.deleteNote(accountId.value, change.noteId.value)
            else change.note?.let { dao.upsert(it.toEntity()) }
        }
    }
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

private fun Note.toEntity() = NoteEntity(accountId.value, id.value, title, body, folderId?.value, updatedAt.toEpochMilli())
private fun NoteEntity.toApi() = Note(NoteId(id), AccountId(accountId), title, body, folderId?.let(::FolderId), Instant.ofEpochMilli(updatedAtMillis))
private fun Note.toOutbox(kind: LocalChangeKind) = OutboxEntity(accountId.value, UUID.randomUUID().toString(), id.value, kind.name, updatedAt.toEpochMilli(), title, body, folderId?.value)
private fun OutboxEntity.toApi(): LocalNoteChange {
    val changeKind = LocalChangeKind.valueOf(kind)
    val note = if (changeKind == LocalChangeKind.UPSERT) Note(NoteId(noteId), AccountId(accountId), title.orEmpty(), body.orEmpty(), folderId?.let(::FolderId), Instant.ofEpochMilli(changedAtMillis)) else null
    return LocalNoteChange(changeId, AccountId(accountId), NoteId(noteId), changeKind, Instant.ofEpochMilli(changedAtMillis), note)
}

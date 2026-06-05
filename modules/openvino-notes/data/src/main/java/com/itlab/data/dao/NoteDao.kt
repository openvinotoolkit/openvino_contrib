package com.itlab.data.dao

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import androidx.room.Update
import com.itlab.data.entity.NoteEntity
import kotlinx.coroutines.flow.Flow

@Dao
interface NoteDao {
    @Query("SELECT * FROM notes WHERE isDeleted = 0 AND userId = :userId ORDER BY updatedAt DESC")
    fun getAllNotesByUserId(userId: String): Flow<List<NoteEntity>>

    @Query("SELECT * FROM notes WHERE id = :noteId AND userId = :userId LIMIT 1")
    suspend fun getNoteByIdAndUser(
        noteId: String,
        userId: String,
    ): NoteEntity?

    @Query(
        "SELECT * FROM notes WHERE folderId = :folderId AND userId = :userId AND isDeleted = 0 ORDER BY updatedAt DESC",
    )
    fun getNotesByFolderAndUser(
        folderId: String,
        userId: String,
    ): Flow<List<NoteEntity>>

    @Query("SELECT * FROM notes WHERE isSynced = 0 AND isDeleted = 0 AND userId = :userId")
    suspend fun getUnsyncedNotes(userId: String): List<NoteEntity>

    @Query("SELECT * FROM notes WHERE isDeleted = 1 AND userId = :userId")
    suspend fun getDeletedNotes(userId: String): List<NoteEntity>

    @Query("UPDATE notes SET isDeleted = 1, isSynced = 0, updatedAt = :timestamp WHERE id = :id AND userId = :userId")
    suspend fun softDeleteById(
        id: String,
        userId: String,
        timestamp: Long = System.currentTimeMillis(),
    )

    @Query(
        "UPDATE notes SET isDeleted = 1, isSynced = 0, updatedAt = :timestamp " +
            "WHERE folderId = :folderId AND userId = :userId AND isDeleted = 0",
    )
    suspend fun softDeleteByFolderId(
        folderId: String,
        userId: String,
        timestamp: Long = System.currentTimeMillis(),
    )

    @Query("DELETE FROM notes WHERE id = :id AND userId = :userId")
    suspend fun hardDeleteById(
        id: String,
        userId: String,
    )

    @Insert
    suspend fun insert(note: NoteEntity)

    @Update
    suspend fun update(note: NoteEntity)

    @Query("UPDATE notes SET isSynced = 0 WHERE id = :noteId AND userId = :userId")
    suspend fun markNoteUnsynced(
        noteId: String,
        userId: String,
    )

    @Query("UPDATE notes SET isSynced = 0 WHERE id IN (:noteIds) AND userId = :userId")
    suspend fun markNotesUnsynced(
        noteIds: List<String>,
        userId: String,
    )

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertAll(notes: List<NoteEntity>)
}

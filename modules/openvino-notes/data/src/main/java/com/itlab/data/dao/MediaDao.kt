package com.itlab.data.dao

import androidx.room.Dao
import androidx.room.Delete
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import androidx.room.Update
import com.itlab.data.entity.MediaEntity
import kotlinx.coroutines.flow.Flow

@Dao
interface MediaDao {
    @Query(
        """
        SELECT media.* FROM media
        INNER JOIN notes ON media.noteId = notes.id
        WHERE media.noteId = :noteId AND media.isDeleted = false AND notes.isDeleted = false
    """,
    )
    suspend fun getMediaForNote(noteId: String): List<MediaEntity>

    @Query("SELECT * FROM media WHERE noteId = :noteId")
    suspend fun getAllMediaRowsForNote(noteId: String): List<MediaEntity>

    @Query("UPDATE media SET isDeleted = true, isSynced = false WHERE noteId = :noteId")
    suspend fun softDeleteByNoteId(noteId: String)

    @Query("UPDATE media SET isDeleted = true, isSynced = false WHERE id IN (:mediaIds)")
    suspend fun softDeleteMediaByIds(mediaIds: List<String>)

    @Query(
        """
    SELECT media.* FROM media
    INNER JOIN notes ON media.noteId = notes.id
    WHERE notes.userId = :userId AND media.isDeleted = false AND notes.isDeleted = false
""",
    )
    fun getAllMediaByUserId(userId: String): Flow<List<MediaEntity>>

    @Query(
        """
    SELECT media.* FROM media
    INNER JOIN notes ON media.noteId = notes.id
    WHERE media.isSynced = false AND media.isDeleted = false AND notes.userId = :userId
""",
    )
    suspend fun getUnsyncedMedia(userId: String): List<MediaEntity>

    @Query(
        """
        SELECT media.* FROM media
        INNER JOIN notes ON media.noteId = notes.id
        WHERE media.isSynced = false AND media.isDeleted = true AND notes.userId = :userId
    """,
    )
    suspend fun getDeletedMediaToSync(userId: String): List<MediaEntity>

    @Delete
    suspend fun hardDelete(media: MediaEntity)

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertAll(mediaList: List<MediaEntity>)

    @Update
    suspend fun update(media: MediaEntity)

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insert(media: MediaEntity)
}

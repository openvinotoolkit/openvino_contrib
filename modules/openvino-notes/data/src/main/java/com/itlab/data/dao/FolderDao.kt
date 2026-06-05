package com.itlab.data.dao

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import androidx.room.Update
import com.itlab.data.entity.FolderEntity
import kotlinx.coroutines.flow.Flow
import kotlin.time.Instant

@Dao
interface FolderDao {
    @Query("SELECT * FROM folders WHERE userId = :userId AND isDeleted = 0 ORDER BY name ASC")
    fun getActiveFoldersByUserId(userId: String): Flow<List<FolderEntity>>

    @Query("SELECT * FROM folders WHERE id = :id AND userId = :userId AND isDeleted = 0")
    suspend fun getFolderByIdAndUser(
        id: String,
        userId: String,
    ): FolderEntity?

    @Query("UPDATE folders SET isDeleted = 1, isSynced = 0, updatedAt = :updatedAt WHERE id = :id AND userId = :userId")
    suspend fun softDeleteById(
        id: String,
        userId: String,
        updatedAt: Instant,
    )

    @Query("UPDATE folders SET name = :name, isSynced = 0, updatedAt = :updatedAt WHERE id = :id AND userId = :userId")
    suspend fun updateName(
        id: String,
        userId: String,
        name: String,
        updatedAt: Instant,
    )

    @Query("SELECT * FROM folders WHERE userId = :userId AND isSynced = 0")
    suspend fun getUnsyncedFolders(userId: String): List<FolderEntity>

    @Query("DELETE FROM folders WHERE id = :id AND userId = :userId")
    suspend fun hardDeleteById(
        id: String,
        userId: String,
    )

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insert(folder: FolderEntity)

    @Update
    suspend fun update(folder: FolderEntity)
}

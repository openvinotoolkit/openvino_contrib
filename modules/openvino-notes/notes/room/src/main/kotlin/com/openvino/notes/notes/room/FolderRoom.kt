// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package com.openvino.notes.notes.room

import androidx.room.Dao
import androidx.room.Entity
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import androidx.room.Transaction
import com.openvino.notes.kernel.AccountKey
import com.openvino.notes.notes.api.Folder
import com.openvino.notes.notes.api.FolderId
import com.openvino.notes.notes.api.FolderRemoteApplyResult
import com.openvino.notes.notes.api.LocalChangeKind
import com.openvino.notes.notes.api.LocalFolderChange
import com.openvino.notes.notes.api.RemoteFolderChange
import com.openvino.notes.notes.api.RemoteRevision
import com.openvino.notes.notes.api.port.FolderRepository
import com.openvino.notes.notes.api.port.FolderSyncPort
import java.time.Instant
import java.util.UUID
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map

@Entity(tableName = "folders", primaryKeys = ["accountKey", "id"])
internal data class FolderEntity(
    val accountKey: String,
    val id: String,
    val name: String,
    val createdAtMillis: Long,
    val updatedAtMillis: Long,
)

@Entity(tableName = "folder_outbox", primaryKeys = ["accountKey", "changeId"])
internal data class FolderOutboxEntity(
    val accountKey: String,
    val changeId: String,
    val folderId: String,
    val kind: String,
    val baseRevision: String?,
    val changedAtMillis: Long,
    val name: String?,
    val createdAtMillis: Long?,
)

@Entity(tableName = "folder_remote_revision", primaryKeys = ["accountKey", "folderId"])
internal data class FolderRemoteRevisionEntity(
    val accountKey: String,
    val folderId: String,
    val revision: String,
)

@Dao
internal interface FolderDao {
    @Query("SELECT * FROM folders WHERE accountKey = :accountKey ORDER BY name COLLATE NOCASE")
    fun observe(accountKey: String): Flow<List<FolderEntity>>

    @Query("SELECT * FROM folders WHERE accountKey = :accountKey AND id = :id")
    suspend fun find(accountKey: String, id: String): FolderEntity?

    @Query("SELECT * FROM folders WHERE accountKey = :accountKey AND name = :name COLLATE NOCASE LIMIT 1")
    suspend fun findByName(accountKey: String, name: String): FolderEntity?

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun upsert(entity: FolderEntity)

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun addOutbox(entity: FolderOutboxEntity)

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun saveRevision(entity: FolderRemoteRevisionEntity)

    @Query("SELECT * FROM folder_remote_revision WHERE accountKey = :accountKey AND folderId = :folderId")
    suspend fun revision(accountKey: String, folderId: String): FolderRemoteRevisionEntity?

    @Query("DELETE FROM folders WHERE accountKey = :accountKey AND id = :id")
    suspend fun deleteFolder(accountKey: String, id: String): Int

    @Query("SELECT COUNT(*) FROM notes WHERE accountKey = :accountKey AND folderId = :folderId")
    suspend fun noteCount(accountKey: String, folderId: String): Int

    @Query("SELECT COUNT(*) FROM folder_outbox WHERE accountKey = :accountKey AND folderId = :folderId")
    suspend fun pendingCount(accountKey: String, folderId: String): Int

    @Query("SELECT * FROM folder_outbox WHERE accountKey = :accountKey ORDER BY changedAtMillis LIMIT :limit")
    suspend fun pending(accountKey: String, limit: Int): List<FolderOutboxEntity>

    @Query("DELETE FROM folder_outbox WHERE accountKey = :accountKey AND changeId IN (:changeIds)")
    suspend fun acknowledge(accountKey: String, changeIds: Set<String>)

    @Transaction
    suspend fun saveLocally(folder: FolderEntity, outbox: FolderOutboxEntity) {
        upsert(folder)
        addOutbox(outbox.copy(baseRevision = revision(folder.accountKey, folder.id)?.revision))
    }

    @Transaction
    suspend fun deleteLocally(accountKey: String, id: String, outbox: FolderOutboxEntity): Boolean {
        if (noteCount(accountKey, id) > 0) return false
        val removed = deleteFolder(accountKey, id) > 0
        if (removed) addOutbox(outbox.copy(baseRevision = revision(accountKey, id)?.revision))
        return removed
    }

    @Transaction
    suspend fun applyRemoteUpsert(folder: FolderEntity, remoteRevision: String): Boolean {
        if (pendingCount(folder.accountKey, folder.id) > 0) return false
        upsert(folder)
        saveRevision(FolderRemoteRevisionEntity(folder.accountKey, folder.id, remoteRevision))
        return true
    }

    @Transaction
    suspend fun applyRemoteTombstone(accountKey: String, folderId: String, remoteRevision: String): Boolean {
        if (pendingCount(accountKey, folderId) > 0 || noteCount(accountKey, folderId) > 0) return false
        deleteFolder(accountKey, folderId)
        saveRevision(FolderRemoteRevisionEntity(accountKey, folderId, remoteRevision))
        return true
    }
}

internal class RoomFolderRepository(private val dao: FolderDao) : FolderRepository, FolderSyncPort {
    override fun observe(accountKey: AccountKey): Flow<List<Folder>> =
        dao.observe(accountKey.value).map { entities -> entities.map(FolderEntity::toApi) }

    override suspend fun find(accountKey: AccountKey, id: FolderId): Folder? =
        dao.find(accountKey.value, id.value)?.toApi()

    override suspend fun findByName(accountKey: AccountKey, name: String): Folder? =
        dao.findByName(accountKey.value, name)?.toApi()

    override suspend fun save(folder: Folder) {
        dao.saveLocally(folder.toEntity(), folder.toOutbox(LocalChangeKind.UPSERT))
    }

    override suspend fun delete(accountKey: AccountKey, id: FolderId): Boolean = dao.deleteLocally(
        accountKey.value,
        id.value,
        FolderOutboxEntity(
            accountKey = accountKey.value,
            changeId = UUID.randomUUID().toString(),
            folderId = id.value,
            kind = LocalChangeKind.DELETE.name,
            baseRevision = null,
            changedAtMillis = System.currentTimeMillis(),
            name = null,
            createdAtMillis = null,
        ),
    )

    override suspend fun pendingFolderChanges(accountKey: AccountKey, limit: Int): List<LocalFolderChange> =
        dao.pending(accountKey.value, limit).map(FolderOutboxEntity::toApi)

    override suspend fun acknowledgeFolderChanges(accountKey: AccountKey, changeIds: Set<String>) {
        if (changeIds.isNotEmpty()) dao.acknowledge(accountKey.value, changeIds)
    }

    override suspend fun applyRemoteFolders(
        accountKey: AccountKey,
        changes: List<RemoteFolderChange>,
    ): List<FolderRemoteApplyResult> = changes.map { change ->
        when (change) {
            is RemoteFolderChange.Malformed -> FolderRemoteApplyResult.RejectedMalformed(
                change.folderId,
                change.diagnosticCode,
            )
            is RemoteFolderChange.Upsert -> applyUpsert(accountKey, change)
            is RemoteFolderChange.Tombstone -> applyTombstone(accountKey, change)
        }
    }

    private suspend fun applyUpsert(
        accountKey: AccountKey,
        change: RemoteFolderChange.Upsert,
    ): FolderRemoteApplyResult {
        if (change.folder.accountKey != accountKey) {
            return FolderRemoteApplyResult.RejectedMalformed(change.folder.id, "folders.remote.account_mismatch")
        }
        return if (dao.applyRemoteUpsert(change.folder.toEntity(), change.revision.value)) {
            FolderRemoteApplyResult.Applied(change.folder.id, change.revision)
        } else {
            FolderRemoteApplyResult.Conflict(
                change.folder.id,
                localRevision(accountKey, change.folder.id),
                change.revision,
            )
        }
    }

    private suspend fun applyTombstone(
        accountKey: AccountKey,
        change: RemoteFolderChange.Tombstone,
    ): FolderRemoteApplyResult = if (
        dao.applyRemoteTombstone(accountKey.value, change.folderId.value, change.revision.value)
    ) {
        FolderRemoteApplyResult.TombstoneApplied(change.folderId, change.revision)
    } else {
        FolderRemoteApplyResult.Conflict(
            change.folderId,
            localRevision(accountKey, change.folderId),
            change.revision,
        )
    }

    private suspend fun localRevision(accountKey: AccountKey, folderId: FolderId): RemoteRevision? =
        dao.revision(accountKey.value, folderId.value)?.revision?.let(::RemoteRevision)
}

private fun Folder.toEntity() = FolderEntity(
    accountKey = accountKey.value,
    id = id.value,
    name = name,
    createdAtMillis = createdAt.toEpochMilli(),
    updatedAtMillis = updatedAt.toEpochMilli(),
)

private fun FolderEntity.toApi() = Folder(
    id = FolderId(id),
    accountKey = AccountKey(accountKey),
    name = name,
    createdAt = Instant.ofEpochMilli(createdAtMillis),
    updatedAt = Instant.ofEpochMilli(updatedAtMillis),
)

private fun Folder.toOutbox(kind: LocalChangeKind) = FolderOutboxEntity(
    accountKey = accountKey.value,
    changeId = UUID.randomUUID().toString(),
    folderId = id.value,
    kind = kind.name,
    baseRevision = null,
    changedAtMillis = updatedAt.toEpochMilli(),
    name = name,
    createdAtMillis = createdAt.toEpochMilli(),
)

private fun FolderOutboxEntity.toApi(): LocalFolderChange {
    val changeKind = LocalChangeKind.valueOf(kind)
    val folder = if (changeKind == LocalChangeKind.UPSERT) {
        Folder(
            id = FolderId(folderId),
            accountKey = AccountKey(accountKey),
            name = requireNotNull(name),
            createdAt = Instant.ofEpochMilli(requireNotNull(createdAtMillis)),
            updatedAt = Instant.ofEpochMilli(changedAtMillis),
        )
    } else {
        null
    }
    return LocalFolderChange(
        changeId = changeId,
        accountKey = AccountKey(accountKey),
        folderId = FolderId(folderId),
        kind = changeKind,
        baseRevision = baseRevision?.let(::RemoteRevision),
        changedAt = Instant.ofEpochMilli(changedAtMillis),
        payload = folder,
    )
}

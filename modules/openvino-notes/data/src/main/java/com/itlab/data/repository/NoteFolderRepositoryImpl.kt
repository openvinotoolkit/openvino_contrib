package com.itlab.data.repository

import com.itlab.data.dao.FolderDao
import com.itlab.data.dao.MediaDao
import com.itlab.data.dao.NoteDao
import com.itlab.data.mapper.NoteFolderMapper
import com.itlab.domain.model.NoteFolder
import com.itlab.domain.repository.NoteFolderRepository
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.map
import kotlin.time.Clock

class NoteFolderRepositoryImpl(
    private val folderDao: FolderDao,
    private val noteDao: NoteDao,
    private val mediaDao: MediaDao,
    private val mapper: NoteFolderMapper,
) : NoteFolderRepository {
    override suspend fun createFolder(folder: NoteFolder): String {
        val entity =
            mapper.toEntity(folder).copy(
                isSynced = false,
                isDeleted = false,
            )
        folderDao.insert(entity)
        return folder.id
    }

    override fun observeFolders(userId: String): Flow<List<NoteFolder>> =
        folderDao.getActiveFoldersByUserId(userId).map { entities ->
            entities.map { mapper.toDomain(it) }
        }

    override suspend fun renameFolder(
        id: String,
        userId: String,
        name: String,
    ) {
        val now = Clock.System.now()
        folderDao.updateName(id = id, userId = userId, name = name, updatedAt = now)
    }

    override suspend fun deleteFolder(
        id: String,
        userId: String,
    ) {
        val now = Clock.System.now()
        val timestamp = now.toEpochMilliseconds()
        val notesInFolder = noteDao.getNotesByFolderAndUser(id, userId).first()
        notesInFolder.forEach { note -> mediaDao.softDeleteByNoteId(note.id) }
        noteDao.softDeleteByFolderId(folderId = id, userId = userId, timestamp = timestamp)
        folderDao.softDeleteById(id = id, userId = userId, updatedAt = now)
    }

    override suspend fun getFolderById(
        id: String,
        userId: String,
    ): NoteFolder? =
        folderDao.getFolderByIdAndUser(id, userId)?.let {
            mapper.toDomain(it)
        }

    override suspend fun updateFolder(folder: NoteFolder) {
        val entity = mapper.toEntity(folder).copy(isSynced = false)
        folderDao.update(entity)
    }
}

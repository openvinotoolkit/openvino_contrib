package com.itlab.domain.repository

import com.itlab.domain.model.NoteFolder
import kotlinx.coroutines.flow.Flow

interface NoteFolderRepository {
    fun observeFolders(userId: String): Flow<List<NoteFolder>>

    suspend fun createFolder(folder: NoteFolder): String

    suspend fun renameFolder(
        id: String,
        userId: String,
        name: String,
    )

    suspend fun deleteFolder(
        id: String,
        userId: String,
    )

    suspend fun getFolderById(
        id: String,
        userId: String,
    ): NoteFolder?

    suspend fun updateFolder(folder: NoteFolder)
}

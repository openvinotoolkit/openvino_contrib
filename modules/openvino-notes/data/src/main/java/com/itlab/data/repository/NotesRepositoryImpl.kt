package com.itlab.data.repository

import com.itlab.data.dao.MediaDao
import com.itlab.data.dao.NoteDao
import com.itlab.data.mapper.NoteMapper
import com.itlab.domain.model.Note
import com.itlab.domain.repository.NotesRepository
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.map

class NotesRepositoryImpl(
    private val noteDao: NoteDao,
    private val mediaDao: MediaDao,
    private val mapper: NoteMapper,
) : NotesRepository {
    override fun observeNotes(userId: String): Flow<List<Note>> =
        combine(
            noteDao.getAllNotesByUserId(userId),
            mediaDao.getAllMediaByUserId(userId), // Предполагаем, что этот метод возвращает Flow<List<MediaEntity>>
        ) { notes, mediaList ->
            notes.map { noteEntity ->
                val associatedMedia = mediaList.filter { it.noteId == noteEntity.id }
                mapper.toDomain(noteEntity, associatedMedia)
            }
        }

    override fun observeNotesByFolder(
        folderId: String,
        userId: String,
    ): Flow<List<Note>> =
        combine(
            noteDao.getNotesByFolderAndUser(folderId, userId),
            mediaDao.getAllMediaByUserId(userId),
        ) { notes, mediaList ->
            notes.map { noteEntity ->
                val associatedMedia = mediaList.filter { it.noteId == noteEntity.id }
                mapper.toDomain(noteEntity, associatedMedia)
            }
        }

    override suspend fun getNoteById(
        id: String,
        userId: String,
    ): Note? {
        val noteEntity = noteDao.getNoteByIdAndUser(id, userId) ?: return null
        val mediaEntities = mediaDao.getMediaForNote(id) // Если метод suspend и возвращает List<MediaEntity>
        return mapper.toDomain(noteEntity, mediaEntities)
    }

    override suspend fun createNote(note: Note): String {
        val (notesEntity, mediaEntities) = mapper.toEntities(note)
        noteDao.insert(notesEntity)
        if (mediaEntities.isNotEmpty()) mediaDao.insertAll(mediaEntities)
        return note.id
    }

    override suspend fun updateNote(note: Note) {
        val (newNoteEntity, incomingMediaEntities) = mapper.toEntities(note)

        noteDao.getNoteByIdAndUser(note.id, note.userId) ?: return

        val localMediaInDb = mediaDao.getMediaForNote(note.id)

        val incomingIds = incomingMediaEntities.map { it.id }.toSet()
        val mediaIdsToSoftDelete =
            localMediaInDb
                .map { it.id }
                .filter { id -> id !in incomingIds }

        if (mediaIdsToSoftDelete.isNotEmpty()) {
            mediaDao.softDeleteMediaByIds(mediaIdsToSoftDelete)
        }

        val finalMediaToInsert =
            incomingMediaEntities.map { incoming ->
                val alreadyExistingMedia = localMediaInDb.find { it.id == incoming.id }
                if (alreadyExistingMedia != null) {
                    incoming.copy(
                        remoteUrl = alreadyExistingMedia.remoteUrl ?: incoming.remoteUrl,
                        localPath = alreadyExistingMedia.localPath ?: incoming.localPath,
                        isSynced =
                            alreadyExistingMedia.isSynced &&
                                incoming.localPath == alreadyExistingMedia.localPath,
                        isDeleted = false,
                    )
                } else {
                    incoming.copy(isSynced = false, isDeleted = false)
                }
            }

        if (finalMediaToInsert.isNotEmpty()) {
            mediaDao.insertAll(finalMediaToInsert)
        }

        val activeMediaIds = mediaDao.getMediaForNote(note.id).map { it.id }.toSet()
        val prunedContent =
            mapper.pruneNoteContentJson(
                contentJson = newNoteEntity.content,
                activeMediaIds = activeMediaIds,
            )
        val noteWithPendingSync =
            newNoteEntity.copy(
                content = prunedContent,
                isSynced = false,
            )
        noteDao.update(noteWithPendingSync)
    }

    override suspend fun deleteNote(
        id: String,
        userId: String,
    ) {
        noteDao.softDeleteById(id, userId)
        mediaDao.softDeleteByNoteId(id)
    }
}

package com.itlab.data.repository

import com.itlab.data.dao.MediaDao
import com.itlab.data.dao.NoteDao
import com.itlab.data.entity.MediaEntity
import com.itlab.data.entity.NoteEntity
import com.itlab.data.mapper.NoteMapper
import com.itlab.domain.model.ContentItem
import com.itlab.domain.model.DataSource
import com.itlab.domain.model.Note
import io.mockk.Runs
import io.mockk.coEvery
import io.mockk.coVerify
import io.mockk.every
import io.mockk.just
import io.mockk.mockk
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.flowOf
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.time.Clock

class NotesRepositoryImplTest {
    private val noteDao = mockk<NoteDao>(relaxed = true)
    private val mediaDao = mockk<MediaDao>(relaxed = true)
    private val mapper = NoteMapper()
    private val repository = NotesRepositoryImpl(noteDao, mediaDao, mapper)
    private val testUserId = "test_user_1"

    private fun createTestNote(
        id: String,
        contentItems: List<ContentItem> = emptyList(),
    ) = Note(
        userId = testUserId,
        id = id,
        title = "Test Title",
        contentItems = contentItems,
        createdAt = Clock.System.now(),
        updatedAt = Clock.System.now(),
    )

    @Test
    fun `createNote inserts note and media if exists`() =
        runTest {
            val note =
                Note(
                    userId = testUserId,
                    id = "note_1",
                    title = "Test",
                    contentItems = emptyList(),
                    createdAt = Clock.System.now(),
                    updatedAt = Clock.System.now(),
                )

            repository.createNote(note)

            coEvery { noteDao.insert(any()) }
            coVerify(exactly = 0) { mediaDao.insertAll(any()) }
        }

    @Test
    fun `updateNote does nothing if note does not exist in database`() =
        runTest {
            val note = createTestNote("note_1")
            coEvery { noteDao.getNoteByIdAndUser("note_1", testUserId) } returns null

            repository.updateNote(note)

            coVerify(exactly = 0) { noteDao.update(any()) }
            coVerify(exactly = 0) { mediaDao.insertAll(any()) }
        }

    @Test
    fun `updateNote correctly soft deletes removed media and inserts new media`() =
        runTest {
            val noteId = "note_123"

            val imageItem =
                ContentItem.Image(
                    source = DataSource(localPath = "some/new_path.png", remoteUrl = null),
                    mimeType = "image/png",
                )
            val updatedNote = createTestNote(noteId, listOf(imageItem))

            coEvery { noteDao.getNoteByIdAndUser(noteId, testUserId) } returns mockk(relaxed = true)

            val oldMediaEntity =
                mockk<MediaEntity>(relaxed = true) {
                    every { id } returns "media_old"
                }

            coEvery { mediaDao.getMediaForNote(noteId) } returns listOf(oldMediaEntity)

            repository.updateNote(updatedNote)

            coVerify(exactly = 1) { noteDao.update(any()) }
            coVerify(exactly = 1) { mediaDao.softDeleteMediaByIds(listOf("media_old")) }
            coVerify(exactly = 1) { mediaDao.insertAll(any()) }
        }

    @Test
    fun `updateNote without changes to media should only update note and not soft delete anything`() =
        runTest {
            val noteId = "note_1"
            val noteWithoutMedia = createTestNote(noteId, emptyList())

            coEvery { noteDao.getNoteByIdAndUser(noteId, testUserId) } returns mockk(relaxed = true)
            coEvery { mediaDao.getMediaForNote(noteId) } returns emptyList()

            repository.updateNote(noteWithoutMedia)

            coVerify(exactly = 1) { noteDao.update(any()) }
            coVerify(exactly = 0) { mediaDao.softDeleteMediaByIds(any()) }
            coVerify(exactly = 0) { mediaDao.insertAll(any()) }
        }

    @Test
    fun `deleteNote deletes by entity from dao`() =
        runTest {
            val noteId = "1"
            coEvery { noteDao.getNoteByIdAndUser(noteId, testUserId) } returns mockk(relaxed = true)

            repository.deleteNote(noteId, testUserId)
            coVerify { noteDao.softDeleteById(noteId, testUserId, any()) }
            coVerify { mediaDao.softDeleteByNoteId(noteId) }
        }

    @Test
    fun `observeNotes emits mapped list from dao`() =
        runTest {
            val entities = listOf(mockk<NoteEntity>(relaxed = true))
            coEvery { noteDao.getAllNotesByUserId(testUserId) } returns flowOf(entities)
            coEvery { mediaDao.getAllMediaByUserId(testUserId) } returns flowOf(emptyList())

            val result = repository.observeNotes(testUserId).first()

            assertEquals(1, result.size)
            coVerify { noteDao.getAllNotesByUserId(testUserId) }
        }

    @Test
    fun `observeNotesByFolder emits filtered list`() =
        runTest {
            val folderId = "folder_x"
            coEvery { noteDao.getNotesByFolderAndUser(folderId, testUserId) } returns flowOf(emptyList())
            coEvery { mediaDao.getAllMediaByUserId(testUserId) } returns flowOf(emptyList())

            val result = repository.observeNotesByFolder(folderId, testUserId).first()

            assertTrue(result.isEmpty())
            coVerify { noteDao.getNotesByFolderAndUser(folderId, testUserId) }
        }

    @Test
    fun `updateNote with media calls insertAll`() =
        runTest {
            val noteId = "note_123"
            val imageItem =
                ContentItem.Image(
                    source = DataSource(localPath = "some/path", remoteUrl = null),
                    mimeType = "image/png",
                )
            val noteWithMedia = createTestNote(noteId, listOf(imageItem))

            coEvery { noteDao.getNoteByIdAndUser(noteId, testUserId) } returns mockk(relaxed = true)
            coEvery { mediaDao.getMediaForNote(noteId) } returns emptyList()

            repository.updateNote(noteWithMedia)

            coVerify(exactly = 1) { noteDao.update(any()) }
            coVerify(exactly = 1) { mediaDao.insertAll(any()) }
            coVerify(exactly = 0) { mediaDao.softDeleteMediaByIds(any()) }
        }

    @Test
    fun `updateNote without media should only call update and delete`() =
        runTest {
            val noteId = "2"
            val noteWithoutMedia =
                Note(
                    userId = testUserId,
                    id = noteId,
                    title = "No Media",
                    contentItems = emptyList(),
                    createdAt = Clock.System.now(),
                    updatedAt = Clock.System.now(),
                )

            coEvery { noteDao.getNoteByIdAndUser(noteId, testUserId) } returns mockk(relaxed = true)
            coEvery { mediaDao.getMediaForNote(noteId) } returns emptyList()

            repository.updateNote(noteWithoutMedia)

            coVerify(exactly = 1) { noteDao.update(any()) }

            coVerify(exactly = 0) { mediaDao.softDeleteMediaByIds(any()) }
            coVerify(exactly = 0) { mediaDao.insertAll(any()) }
        }

    @Test
    fun `deleteNote does nothing if note not found`() =
        runTest {
            val nonExistentId = "non_existent"

            repository.deleteNote(nonExistentId, testUserId)

            coVerify(exactly = 1) { noteDao.softDeleteById(nonExistentId, testUserId, any()) }
        }

    @Test
    fun `getNoteById returns null correctly`() =
        runTest {
            coEvery { noteDao.getNoteByIdAndUser("any", testUserId) } returns null
            val result = repository.getNoteById("any", testUserId)
            assertNull(result)
        }

    @Test
    fun `deleteNote should not call dao delete if note is null`() =
        runTest {
            val missingId = "missing_id"

            repository.deleteNote(missingId, testUserId)

            coVerify(exactly = 1) { noteDao.softDeleteById(missingId, testUserId, any()) }
        }

    @Test
    fun `observeNotesByFolder emits empty list and then content`() =
        runTest {
            val folderId = "folder_1"
            val flow = MutableStateFlow<List<NoteEntity>>(emptyList())
            coEvery { noteDao.getNotesByFolderAndUser(folderId, testUserId) } returns flow
            coEvery { mediaDao.getAllMediaByUserId(testUserId) } returns flowOf(emptyList())

            val firstResult = repository.observeNotesByFolder(folderId, testUserId).first()
            assertTrue(firstResult.isEmpty())

            val entity =
                mockk<NoteEntity>(relaxed = true) {
                    every { id } returns "n1"
                }
            flow.value = listOf(entity)
            val secondResult = repository.observeNotesByFolder(folderId, testUserId).first()
            assertEquals(1, secondResult.size)
        }

    @Test
    fun `observeNotes emits list when dao has data`() =
        runTest {
            val entity = mockk<NoteEntity>(relaxed = true)
            coEvery { noteDao.getAllNotesByUserId(testUserId) } returns flowOf(listOf(entity))
            coEvery { mediaDao.getAllMediaByUserId(testUserId) } returns flowOf(emptyList())

            val result = repository.observeNotes(testUserId).first()

            assertEquals(1, result.size)
        }

    @Test
    fun `deleteNote handles missing note gracefully`() =
        runTest {
            val unknownId = "unknown"

            repository.deleteNote(unknownId, testUserId)

            coVerify(exactly = 1) { noteDao.softDeleteById(unknownId, testUserId, any()) }
        }

    @Test
    fun `getNoteById returns mapped domain note when entity exists`() =
        runTest {
            val noteId = "note_123"
            val entity = mockk<NoteEntity>(relaxed = true)

            coEvery { noteDao.getNoteByIdAndUser(noteId, testUserId) } returns entity
            coEvery { mediaDao.getMediaForNote(noteId) } returns emptyList()

            val result = repository.getNoteById(noteId, testUserId)

            assertNotNull(result)
        }

    @Test
    fun `saveNote inserts media entities when note has media`() =
        runTest {
            val imageItem =
                ContentItem.Image(
                    source = DataSource(localPath = "local/path.jpg", remoteUrl = null),
                    mimeType = "image/jpeg",
                )
            val noteWithMedia =
                Note(
                    userId = testUserId,
                    id = "note_with_pic",
                    title = "Vacation",
                    contentItems = listOf(imageItem),
                )

            coEvery { noteDao.insert(any()) } just Runs
            coEvery { mediaDao.insertAll(any()) } just Runs
            coEvery { noteDao.getNoteByIdAndUser("note_with_pic", testUserId) } returns mockk(relaxed = true)

            repository.updateNote(noteWithMedia)

            coVerify(exactly = 1) { mediaDao.insertAll(any()) }
        }
}

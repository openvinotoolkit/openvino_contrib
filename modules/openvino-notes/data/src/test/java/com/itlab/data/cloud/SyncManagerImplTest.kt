package com.itlab.data.cloud

import android.content.Context
import com.itlab.data.dao.FolderDao
import com.itlab.data.dao.MediaDao
import com.itlab.data.dao.NoteDao
import com.itlab.data.entity.FolderEntity
import com.itlab.data.entity.MediaEntity
import com.itlab.data.entity.NoteEntity
import com.itlab.data.mapper.FolderEntityJsonConverter
import com.itlab.data.mapper.NoteEntityJsonConverter
import com.itlab.data.mapper.NoteMapper
import com.itlab.domain.cloud.CloudDataSource
import com.itlab.domain.cloud.CloudMediaMetadata
import com.itlab.domain.cloud.Result
import com.itlab.domain.cloud.SyncState
import io.mockk.MockKAnnotations
import io.mockk.coEvery
import io.mockk.coVerify
import io.mockk.coVerifyOrder
import io.mockk.every
import io.mockk.impl.annotations.MockK
import io.mockk.mockk
import io.mockk.unmockkAll
import kotlinx.coroutines.flow.flowOf
import kotlinx.coroutines.runBlocking
import org.junit.After
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import java.io.File
import java.io.IOException
import kotlin.coroutines.cancellation.CancellationException
import kotlin.test.assertEquals
import kotlin.time.Clock

class SyncManagerImplTest {
    @MockK(relaxed = true)
    lateinit var noteDao: NoteDao

    @MockK(relaxed = true)
    lateinit var mediaDao: MediaDao

    @MockK(relaxed = true)
    lateinit var folderDao: FolderDao

    @MockK(relaxed = true)
    lateinit var cloudDataSource: CloudDataSource

    @MockK(relaxed = true)
    lateinit var folderConverter: FolderEntityJsonConverter

    @MockK(relaxed = true)
    lateinit var jsonConverter: NoteEntityJsonConverter

    @MockK(relaxed = true)
    lateinit var noteMapper: NoteMapper

    @MockK(relaxed = true)
    lateinit var context: Context

    private lateinit var pusher: SyncPusher
    private lateinit var puller: SyncPuller
    private lateinit var cleaner: SyncCleaner
    private lateinit var syncManager: SyncManagerImpl

    private val userId = "user1"
    private val now = Clock.System.now()
    private lateinit var tempMediaFile: File

    @Before
    fun setUp() {
        MockKAnnotations.init(this)

        tempMediaFile = File.createTempFile("test_media", ".png")

        val daos = SyncDaoContainer(noteDao, folderDao, mediaDao)
        val mappers = SyncMappers(jsonConverter, folderConverter, noteMapper)

        pusher = SyncPusher(daos, mappers, cloudDataSource)
        puller = SyncPuller(daos, mappers, cloudDataSource, context)
        cleaner = SyncCleaner(folderDao, noteDao, mediaDao)

        syncManager = SyncManagerImpl(pusher, puller, cleaner)

        every { context.filesDir } returns tempMediaFile.parentFile

        every { folderDao.getActiveFoldersByUserId(any()) } returns flowOf(emptyList())
        every { noteDao.getAllNotesByUserId(any()) } returns flowOf(emptyList())
        every { mediaDao.getAllMediaByUserId(any()) } returns flowOf(emptyList())
        coEvery { mediaDao.getMediaForNote(any()) } returns emptyList()
        coEvery { mediaDao.getDeletedMediaToSync(any()) } returns emptyList()
        every { noteMapper.pruneNoteContentJson(any(), any()) } answers { firstArg() }
    }

    @After
    fun tearDown() {
        if (tempMediaFile.exists()) {
            tempMediaFile.delete()
        }
        unmockkAll()
    }

    private fun createTestNote(id: String = "n1") =
        NoteEntity(
            id = id,
            userId = userId,
            title = "T",
            content = "C",
            createdAt = now,
            updatedAt = now,
            isSynced = false,
        )

    private fun createTestMedia(id: String = "m1") =
        MediaEntity(
            id = id,
            noteId = "n1",
            type = "IMAGE",
            localPath = tempMediaFile.absolutePath,
            remoteUrl = "url",
            mimeType = "image/png",
            size = 100,
            isSynced = false,
        )

    private fun createTestFolder(id: String = "f1") =
        FolderEntity(
            id = id,
            userId = userId,
            name = "F",
            createdAt = now,
            updatedAt = now,
            isSynced = false,
            isDeleted = false,
            metadata = emptyMap(),
        )

    @Test
    fun `pusher should upload note when unsynced`() =
        runBlocking {
            val note = createTestNote()
            coEvery { noteDao.getUnsyncedNotes(userId) } returns listOf(note)
            coEvery { noteDao.getNoteByIdAndUser(note.id, userId) } returns note
            coEvery { mediaDao.getUnsyncedMedia(userId) } returns emptyList()
            every { with(jsonConverter) { note.toJson() } } returns "{}"
            coEvery { cloudDataSource.uploadNote(any(), any()) } returns Result.Success(Unit)

            pusher.pushChanges(userId)

            coVerify { cloudDataSource.uploadNote("users/$userId/notes/${note.id}", "{}") }
            coVerify { noteDao.update(any()) }
        }

    @Test
    fun `pusher should upload media when unsynced`() =
        runBlocking {
            val media = createTestMedia()
            val note = createTestNote(id = media.noteId)
            coEvery { mediaDao.getUnsyncedMedia(userId) } returns listOf(media)
            coEvery { noteDao.getUnsyncedNotes(userId) } returns listOf(note)
            coEvery { noteDao.getNoteByIdAndUser(note.id, userId) } returns note
            coEvery { cloudDataSource.uploadMedia(any(), any(), any()) } returns Result.Success(Unit)
            every { with(jsonConverter) { note.toJson() } } returns "{}"
            coEvery { cloudDataSource.uploadNote(any(), any()) } returns Result.Success(Unit)

            pusher.pushChanges(userId)

            coVerify {
                cloudDataSource.uploadMedia(
                    "users/$userId/media/${media.noteId}_${media.id}",
                    any(),
                    media.mimeType,
                )
            }
            coVerify { mediaDao.update(any()) }
            coVerify { noteDao.markNotesUnsynced(listOf(media.noteId), userId) }
            coVerify(atLeast = 1) { cloudDataSource.uploadNote("users/$userId/notes/${note.id}", "{}") }
        }

    @Test
    fun `puller should download note and insert into dao`() =
        runBlocking {
            val remoteKey = "users/$userId/notes/n1"
            coEvery { cloudDataSource.listNoteMetadata(userId) } returns
                Result.Success(
                    listOf(
                        mockk {
                            every { key } returns remoteKey
                            every { updatedAt } returns now
                        },
                    ),
                )
            coEvery { cloudDataSource.downloadNote(remoteKey) } returns Result.Success("{}")

            val testNote = createTestNote()
            every { jsonConverter.toEntity("{}", userId) } returns testNote

            coEvery { cloudDataSource.listFolderMetadata(userId) } returns Result.Success(emptyList())
            coEvery { cloudDataSource.listMediaMetadata(userId) } returns Result.Success(emptyList())
            coEvery { noteDao.getAllNotesByUserId(userId) } returns flowOf(emptyList())

            puller.pullUpdates(userId)

            coVerify { noteDao.insert(testNote) }
        }

    @Test
    fun `puller should download media and insert into dao`() =
        runBlocking {
            val remoteMediaMetadata =
                mockk<CloudMediaMetadata> {
                    every { mediaId } returns "n1_m1"
                    every { key } returns "users/$userId/media/n1_m1"
                    every { mimeType } returns "image/png"
                }

            coEvery { cloudDataSource.listFolderMetadata(userId) } returns Result.Success(emptyList())
            coEvery { cloudDataSource.listNoteMetadata(userId) } returns Result.Success(emptyList())
            coEvery { cloudDataSource.listMediaMetadata(userId) } returns Result.Success(listOf(remoteMediaMetadata))
            coEvery { mediaDao.getAllMediaByUserId(userId) } returns flowOf(emptyList())
            coEvery { cloudDataSource.downloadMedia(any(), any()) } returns Result.Success(Unit)

            puller.pullUpdates(userId)

            coVerify { mediaDao.insert(any()) }
        }

    @Test
    fun `sync should update state to Error on exception`() =
        runBlocking {
            coEvery { folderDao.getUnsyncedFolders(userId) } throws IOException("Network Error")

            val result = runCatching { syncManager.sync(userId) }

            assertTrue(result.isFailure)
            assertTrue(syncManager.syncState.value is SyncState.Error)
        }

    @Test
    fun `cleaner should delete folder not present in remote`() =
        runBlocking {
            val localFolder = createTestFolder(id = "f1")
            every { folderDao.getActiveFoldersByUserId(userId) } returns flowOf(listOf(localFolder))

            cleaner.cleanMissingFoldersLocally(userId, emptySet())

            coVerify { folderDao.hardDeleteById("f1", userId) }
        }

    @Test
    fun `sync should call pusher and puller in correct order`() =
        runBlocking {
            val mockPusher = mockk<SyncPusher>(relaxed = true)
            val mockPuller = mockk<SyncPuller>()
            val mockCleaner = mockk<SyncCleaner>(relaxed = true)

            coEvery { mockPuller.pullUpdates(userId) } returns Triple(emptySet(), emptySet(), emptySet())

            val manager = SyncManagerImpl(mockPusher, mockPuller, mockCleaner)

            manager.sync(userId)

            coVerifyOrder {
                mockPusher.pushChanges(userId)
                mockPuller.pullUpdates(userId)
            }
            assertEquals(SyncState.Success, manager.syncState.value)
        }

    @Test
    fun `pullUpdates should orchestrate puller and cleaner in correct order`() =
        runBlocking {
            val mockPusher = mockk<SyncPusher>(relaxed = true)
            val mockPuller = mockk<SyncPuller>()
            val mockCleaner = mockk<SyncCleaner>(relaxed = true)
            val manager = SyncManagerImpl(mockPusher, mockPuller, mockCleaner)

            val folders = setOf("f1")
            val notes = setOf("n1")
            val media = setOf("m1")
            coEvery { mockPuller.pullUpdates(userId) } returns Triple(folders, notes, media)

            manager.pullUpdates(userId)

            coVerifyOrder {
                mockPuller.pullUpdates(userId)
                mockCleaner.cleanMissingMediaLocally(userId, media)
                mockCleaner.cleanMissingNotesLocally(userId, notes)
                mockCleaner.cleanMissingFoldersLocally(userId, folders)
            }
        }

    @Test
    fun `pusher should delete folder when isDeleted is true`() =
        runBlocking {
            val deletedFolder = createTestFolder(id = "f1").copy(isDeleted = true)
            coEvery { folderDao.getUnsyncedFolders(userId) } returns listOf(deletedFolder)
            coEvery { cloudDataSource.deleteObject(any()) } returns Result.Success(Unit)

            pusher.pushChanges(userId)

            coVerify { cloudDataSource.deleteObject("users/$userId/folders/f1") }
            coVerify { folderDao.hardDeleteById("f1", userId) }
        }

    @Test
    fun `pusher should throw exception when cloud returns Error`() =
        runBlocking {
            val note = createTestNote()
            coEvery { noteDao.getUnsyncedNotes(userId) } returns listOf(note)
            every { with(jsonConverter) { note.toJson() } } returns "{}"
            coEvery { cloudDataSource.uploadNote(any(), any()) } returns Result.Error(Exception("Server 500"))

            val result = runCatching { pusher.pushChanges(userId) }

            assertTrue(result.isFailure)
        }

    @Test(expected = CancellationException::class)
    fun `sync should rethrow CancellationException`() =
        runBlocking {
            coEvery { folderDao.getUnsyncedFolders(userId) } throws CancellationException()

            syncManager.sync(userId)
        }

    @Test
    fun `pusher should upload folder and update state when not deleted`() =
        runBlocking {
            val folder = createTestFolder()
            coEvery { folderDao.getUnsyncedFolders(userId) } returns listOf(folder)
            every { with(folderConverter) { folder.toJson() } } returns "{}"
            coEvery { cloudDataSource.uploadFolder(any(), any()) } returns Result.Success(Unit)

            pusher.pushChanges(userId)

            coVerify { cloudDataSource.uploadFolder("users/$userId/folders/${folder.id}", "{}") }
            coVerify { folderDao.update(folder.copy(isSynced = true)) }
        }

    @Test
    fun `pusher should delete remote and local note when in deletedNotes`() =
        runBlocking {
            val note = createTestNote()
            coEvery { noteDao.getDeletedNotes(userId) } returns listOf(note)
            coEvery { cloudDataSource.deleteObject(any()) } returns Result.Success(Unit)

            pusher.pushChanges(userId)

            coVerify { cloudDataSource.deleteObject("users/$userId/notes/${note.id}") }
            coVerify { noteDao.hardDeleteById(note.id, userId) }
        }

    @Test
    fun `pusher should delete remote media and local file when in deletedMedia`() =
        runBlocking {
            val media = createTestMedia()
            coEvery { mediaDao.getDeletedMediaToSync(userId) } returns listOf(media)
            coEvery { cloudDataSource.deleteObject(any()) } returns Result.Success(Unit)

            assertTrue(File(media.localPath!!).exists())

            pusher.pushChanges(userId)

            coVerify { cloudDataSource.deleteObject("users/$userId/media/${media.noteId}_${media.id}") }
            coVerify { mediaDao.hardDelete(media) }

            assertTrue(!File(media.localPath!!).exists())
        }

    @Test
    fun `pusher should skip media upload if local file does not exist`() =
        runBlocking {
            val media = createTestMedia().copy(localPath = "/non/existent/path/file.png")
            coEvery { mediaDao.getUnsyncedMedia(userId) } returns listOf(media)

            pusher.pushChanges(userId)

            coVerify(exactly = 0) { cloudDataSource.uploadMedia(any(), any(), any()) }
        }

    @Test
    fun `puller should insert media with type FILE when mimeType is not image`() =
        runBlocking {
            val remoteMediaMetadata =
                mockk<CloudMediaMetadata> {
                    every { mediaId } returns "n1_m2"
                    every { key } returns "users/$userId/media/n1_m2"
                    every { mimeType } returns "application/pdf" // Не image/*
                }

            coEvery { cloudDataSource.listFolderMetadata(userId) } returns Result.Success(emptyList())
            coEvery { cloudDataSource.listNoteMetadata(userId) } returns Result.Success(emptyList())
            coEvery { cloudDataSource.listMediaMetadata(userId) } returns Result.Success(listOf(remoteMediaMetadata))
            coEvery { mediaDao.getAllMediaByUserId(userId) } returns flowOf(emptyList())
            coEvery { cloudDataSource.downloadMedia(any(), any()) } returns Result.Success(Unit)

            puller.pullUpdates(userId)

            coVerify {
                mediaDao.insert(
                    withArg {
                        assertEquals("FILE", it.type)
                        assertEquals("m2", it.id)
                    },
                )
            }
        }

    @Test
    fun `cleaner should delete missing notes locally`() =
        runBlocking {
            val localNote = createTestNote(id = "n1")
            every { noteDao.getAllNotesByUserId(userId) } returns flowOf(listOf(localNote))

            cleaner.cleanMissingNotesLocally(userId, emptySet())

            coVerify { noteDao.hardDeleteById("n1", userId) }
        }

    @Test
    fun `cleaner should delete missing media and local file locally`() =
        runBlocking {
            val localMedia = createTestMedia(id = "m1")
            every { mediaDao.getAllMediaByUserId(userId) } returns flowOf(listOf(localMedia))

            assertTrue(File(localMedia.localPath!!).exists())

            cleaner.cleanMissingMediaLocally(userId, emptySet())

            coVerify { mediaDao.hardDelete(localMedia) }

            assertTrue(!File(localMedia.localPath!!).exists())
        }

    @Test
    fun `sync should catch pullException but keep sync successful for WorkManager`() =
        runBlocking {
            val mockPusher = mockk<SyncPusher>(relaxed = true)
            val mockPuller = mockk<SyncPuller>()
            val mockCleaner = mockk<SyncCleaner>(relaxed = true)
            val manager = SyncManagerImpl(mockPusher, mockPuller, mockCleaner)

            coEvery { mockPuller.pullUpdates(userId) } throws IOException("No internet connection")

            val result = runCatching { manager.sync(userId) }

            assertTrue(result.isSuccess)
            assertTrue(manager.syncState.value is SyncState.Error)

            assertEquals(
                "Pull failed, but changes were pushed successfully",
                (manager.syncState.value as SyncState.Error).message,
            )
        }
}

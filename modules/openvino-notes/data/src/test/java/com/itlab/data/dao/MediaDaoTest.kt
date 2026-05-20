package com.itlab.data.dao

import androidx.room.Room
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.itlab.data.db.AppDatabase
import com.itlab.data.entity.MediaEntity
import com.itlab.data.entity.NoteEntity
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.test.runTest
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.annotation.Config
import kotlin.time.Instant

@RunWith(AndroidJUnit4::class)
@Config(manifest = Config.NONE, sdk = [34])
class MediaDaoTest {
    private lateinit var database: AppDatabase
    private lateinit var mediaDao: MediaDao
    private lateinit var noteDao: NoteDao
    private val testUserId = "test_user_1"
    val testTime = Instant.parse("2026-03-24T12:00:00Z")

    private suspend fun insertParentNote(id: String) {
        val note =
            com.itlab.data.entity.NoteEntity(
                id = id,
                title = "Parent Note",
                content = "Content",
                createdAt = testTime,
                updatedAt = Instant.fromEpochMilliseconds(0),
                isSynced = true,
                userId = testUserId,
            )
        noteDao.insert(note)
    }

    private val defaultPath = """C:\Users\egoru\Downloads\Blazhin_-_Ne_perebivajj_64351892.mp3"""

    private val baseMediaTemplate =
        MediaEntity(
            id = "default_id",
            noteId = "default_note",
            type = "IMAGE",
            localPath = "local/path/to/media.mp3",
            remoteUrl = null,
            mimeType = "audio/mpeg",
            isSynced = false,
            isDeleted = false,
        )

    private fun createMedia(
        id: String,
        noteId: String,
        configure: (MediaEntity.() -> MediaEntity)? = null,
    ): MediaEntity {
        val media = baseMediaTemplate.copy(id = id, noteId = noteId)
        return configure?.invoke(media) ?: media
    }

    @Before
    fun setup() {
        database =
            Room
                .inMemoryDatabaseBuilder(
                    ApplicationProvider.getApplicationContext(),
                    AppDatabase::class.java,
                ).allowMainThreadQueries()
                .build()

        mediaDao = database.mediaDao()
        noteDao = database.noteDao()
    }

    @After
    fun cleanup() {
        database.close()
    }

    @Test
    fun `insert and getMediaForNote should return correct media with paths`() =
        runTest {
            val noteId = "note1"
            insertParentNote(noteId)

            val audioPath = defaultPath
            val media = createMedia(id = "m1", noteId = "note1") { copy(localPath = audioPath) }

            mediaDao.insert(media)

            val result = mediaDao.getMediaForNote("note1")

            assertEquals(1, result.size)
            assertEquals(audioPath, result[0].localPath)
            assertEquals("audio/mpeg", result[0].mimeType)
        }

    @Test
    fun `insertAll should handle list of media and replace on conflict`() =
        runTest {
            val noteId = "note1"
            insertParentNote(noteId)

            val list =
                listOf(
                    createMedia("m1", "note1"),
                    createMedia("m2", "note1"),
                )

            mediaDao.insertAll(list)

            val updatedMedia =
                createMedia("m1", "note1") {
                    copy(remoteUrl = "https://s3.yandex.net/bucket/audio.mp3")
                }
            mediaDao.insertAll(listOf(updatedMedia))

            val result = mediaDao.getMediaForNote("note1")
            val m1 = result.find { it.id == "m1" }

            assertEquals(2, result.size)
            assertEquals("https://s3.yandex.net/bucket/audio.mp3", m1?.remoteUrl)
        }

    @Test
    fun `softDeleteMediaByIds should mark specific media as deleted and hide from active queries`() =
        runTest {
            val noteId = "note1"
            insertParentNote(noteId)

            mediaDao.insertAll(
                listOf(
                    createMedia("m1", noteId),
                    createMedia("m2", noteId),
                ),
            )

            mediaDao.softDeleteMediaByIds(listOf("m1"))

            val activeResult = mediaDao.getMediaForNote(noteId)
            assertEquals(1, activeResult.size)
            assertEquals("m2", activeResult[0].id)
        }

    @Test
    fun `softDeleteByNoteId should mark all media associated with note as deleted`() =
        runTest {
            insertParentNote("note1")
            insertParentNote("note2")

            mediaDao.insertAll(
                listOf(
                    createMedia("m1", "note1"),
                    createMedia("m2", "note1"),
                    createMedia("m3", "note2"),
                ),
            )

            mediaDao.softDeleteByNoteId("note1")

            val mediaNote1 = mediaDao.getMediaForNote("note1")
            val mediaNote2 = mediaDao.getMediaForNote("note2")

            assertTrue(mediaNote1.isEmpty())
            assertEquals(1, mediaNote2.size)
        }

    @Test
    fun `getUnsyncedMedia should return only media with isSynced false`() =
        runTest {
            insertParentNote("note1")

            val syncedMedia = createMedia("m1", "note1").copy(isSynced = true)
            val unsyncedMedia = createMedia("m2", "note1").copy(isSynced = false)

            mediaDao.insert(syncedMedia)
            mediaDao.insert(unsyncedMedia)

            val result = mediaDao.getUnsyncedMedia(testUserId)

            assertEquals(1, result.size)
            assertEquals("m2", result[0].id)
            assertEquals(false, result[0].isSynced)
        }

    @Test
    fun `getDeletedMediaToSync should return only unsynced media marked as deleted`() =
        runTest {
            insertParentNote("note1")

            val activeUnsynced = createMedia("m1", "note1") { copy(isSynced = false, isDeleted = false) }
            val deletedUnsynced = createMedia("m2", "note1") { copy(isSynced = false, isDeleted = true) }
            val deletedSynced = createMedia("m3", "note1") { copy(isSynced = true, isDeleted = true) }

            mediaDao.insertAll(listOf(activeUnsynced, deletedUnsynced, deletedSynced))

            val result = mediaDao.getDeletedMediaToSync(testUserId)

            assertEquals(1, result.size)
            assertEquals("m2", result[0].id)
        }

    @Test
    fun `update should change sync status and remote url`() =
        runTest {
            insertParentNote("note1")
            val media = createMedia("m1", "note1").copy(isSynced = false, remoteUrl = null)
            mediaDao.insert(media)

            val updatedMedia = media.copy(isSynced = true, remoteUrl = "cloud://path/to/file")
            mediaDao.update(updatedMedia)

            val result = mediaDao.getMediaForNote("note1")[0]

            assertTrue(result.isSynced)
            assertEquals("cloud://path/to/file", result.remoteUrl)
        }

    @Test
    fun `getAllMedia should return all media from different notes`() =
        runTest {
            insertParentNote("note1")
            insertParentNote("note2")

            val media1 = createMedia("m1", "note1")
            val media2 = createMedia("m2", "note2")

            mediaDao.insert(media1)
            mediaDao.insert(media2)

            val result = mediaDao.getAllMediaByUserId(testUserId).first()

            assertEquals(2, result.size)
            assertTrue(result.any { it.id == "m1" })
            assertTrue(result.any { it.id == "m2" })
        }

    @Test
    fun `getAllMedia flow should emit new list when data changes`() =
        runTest {
            insertParentNote("note1")

            val result1 = mediaDao.getAllMediaByUserId(testUserId).first()
            assertTrue(result1.isEmpty())

            mediaDao.insert(createMedia("m1", "note1"))

            val result2 = mediaDao.getAllMediaByUserId(testUserId).first()
            assertEquals(1, result2.size)
            assertEquals("m1", result2[0].id)
        }

    @Test
    fun `getUnsyncedMedia should isolate data and return only media belonging to requested userId`() =
        runTest {
            val otherUserId = "stranger_danger"
            insertParentNote("note_current_user")

            val otherNote =
                NoteEntity(
                    id = "note_other_user",
                    title = "Other User Note",
                    content = "Secret Content",
                    createdAt = testTime,
                    updatedAt = Instant.fromEpochMilliseconds(0),
                    isSynced = true,
                    userId = otherUserId,
                )
            noteDao.insert(otherNote)

            val currentUserMedia = createMedia("media_my", "note_current_user").copy(isSynced = false)
            val otherUserMedia = createMedia("media_alien", "note_other_user").copy(isSynced = false)

            mediaDao.insert(currentUserMedia)
            mediaDao.insert(otherUserMedia)

            val result = mediaDao.getUnsyncedMedia(testUserId)

            assertEquals(1, result.size)
            assertEquals("media_my", result[0].id)
        }

    @Test
    fun `hardDelete should physically remove row from database`() =
        runTest {
            insertParentNote("note1")
            val media = createMedia("m1", "note1")
            mediaDao.insert(media)

            mediaDao.hardDelete(media)

            val result = mediaDao.getMediaForNote("note1")
            assertTrue(result.isEmpty())
        }
}

package com.itlab.data

import android.content.Context
import androidx.room.Room
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.itlab.data.dao.MediaDao
import com.itlab.data.dao.NoteDao
import com.itlab.data.db.AppDatabase
import com.itlab.data.entity.MediaEntity
import com.itlab.data.entity.NoteEntity
import kotlinx.coroutines.runBlocking
import kotlinx.datetime.Clock
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class NoteAndMediaCascadeTest {
    private lateinit var db: AppDatabase
    private lateinit var noteDao: NoteDao
    private lateinit var mediaDao: MediaDao

    @Before
    fun createDb() {
        val context = ApplicationProvider.getApplicationContext<Context>()

        db =
            Room
                .inMemoryDatabaseBuilder(
                    ApplicationProvider.getApplicationContext(),
                    AppDatabase::class.java,
                ).build()
        noteDao = db.noteDao()
        mediaDao = db.mediaDao()
    }

    @After
    fun closeDb() {
        db.close()
    }

    @Test
    fun verifyCascadeDelete_whenNoteDeleted_mediaIsAlsoDeleted() =
        runBlocking {
            val note =
                NoteEntity(
                    id = "note_1",
                    title = "Test",
                    content = "Test",
                    userId = "user-123",
                    folderId = null,
                    createdAt = Clock.System.now(),
                    updatedAt = Clock.System.now(),
                    tags = null,
                    isFavorite = false,
                    isSynced = false,
                    isDeleted = false,
                    summary = null,
                )
            val media =
                MediaEntity(
                    id = "media_1",
                    noteId = "note_1",
                    type = "image",
                    remoteUrl = "",
                    localPath = null,
                    mimeType = "image/jpeg",
                )

            noteDao.insert(note)
            mediaDao.insert(media)

            val mediaBeforeDelete = mediaDao.getMediaForNote("note_1")
            assertEquals(1, mediaBeforeDelete.size)

            noteDao.delete(note)

            val mediaAfterDelete = mediaDao.getMediaForNote("note_1")
            assertTrue(mediaAfterDelete.isEmpty())
        }
}

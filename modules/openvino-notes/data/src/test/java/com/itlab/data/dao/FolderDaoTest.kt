package com.itlab.data.dao

import android.content.Context
import androidx.room.Room
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.itlab.data.db.AppDatabase
import com.itlab.data.entity.FolderEntity
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.test.runTest
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.annotation.Config
import kotlin.time.Clock
import kotlin.time.Instant

@RunWith(AndroidJUnit4::class)
@Config(sdk = [33])
class FolderDaoTest {
    private lateinit var database: AppDatabase
    private lateinit var folderDao: FolderDao

    private val testUserId = "user_1"
    private val otherUserId = "user_2"

    @Before
    fun createDb() {
        val context = ApplicationProvider.getApplicationContext<Context>()
        database =
            Room
                .inMemoryDatabaseBuilder(context, AppDatabase::class.java)
                .allowMainThreadQueries()
                .build()
        folderDao = database.folderDao()
    }

    @After
    fun closeDb() {
        database.close()
    }

    @Test
    fun getActiveFoldersByUserId_returnsOnlyActiveFoldersForSpecificUser() =
        runTest {
            val now = Clock.System.now()

            val folder1 = createTestFolder(id = "1", userId = testUserId, name = "B_Folder", isDeleted = false)
            val folder2 = createTestFolder(id = "2", userId = testUserId, name = "A_Folder", isDeleted = false)
            val deletedFolder = createTestFolder(id = "3", userId = testUserId, name = "C_Folder", isDeleted = true)
            val otherUserFolder = createTestFolder(id = "4", userId = otherUserId, name = "D_Folder", isDeleted = false)

            folderDao.insert(folder1)
            folderDao.insert(folder2)
            folderDao.insert(deletedFolder)
            folderDao.insert(otherUserFolder)

            val activeFolders = folderDao.getActiveFoldersByUserId(testUserId).first()

            assertEquals(2, activeFolders.size)
            assertEquals("A_Folder", activeFolders[0].name)
            assertEquals("B_Folder", activeFolders[1].name)
        }

    @Test
    fun getFolderByIdAndUser_returnsFolderOnlyIfActiveAndBelongsToUser() =
        runTest {
            val folder = createTestFolder(id = "1", userId = testUserId, isDeleted = false)
            val deletedFolder = createTestFolder(id = "2", userId = testUserId, isDeleted = true)

            folderDao.insert(folder)
            folderDao.insert(deletedFolder)

            val found = folderDao.getFolderByIdAndUser("1", testUserId)
            assertNotNull(found)
            assertEquals("1", found?.id)

            val foundByWrongUser = folderDao.getFolderByIdAndUser("1", otherUserId)
            assertNull(foundByWrongUser)

            val foundDeleted = folderDao.getFolderByIdAndUser("2", testUserId)
            assertNull(foundDeleted)
        }

    @Test
    fun softDeleteById_updatesFlagsAndTimestampCorrectly() =
        runTest {
            val folder = createTestFolder(id = "1", userId = testUserId, isSynced = true, isDeleted = false)
            folderDao.insert(folder)

            val currentMillis = System.currentTimeMillis()
            val deleteTime = Instant.fromEpochMilliseconds(currentMillis)

            folderDao.softDeleteById(id = "1", userId = testUserId, updatedAt = deleteTime)

            val updatedList = folderDao.getUnsyncedFolders(testUserId)
            assertEquals(1, updatedList.size)

            val updated = updatedList[0]
            assertTrue(updated.isDeleted)
            assertTrue(!updated.isSynced)

            assertEquals(deleteTime.toEpochMilliseconds(), updated.updatedAt.toEpochMilliseconds())
        }

    @Test
    fun updateName_updatesNameAndResetsSyncFlag() =
        runTest {
            val folder = createTestFolder(id = "1", userId = testUserId, name = "Old Name", isSynced = true)
            folderDao.insert(folder)

            val currentMillis = System.currentTimeMillis()
            val updateTime = Instant.fromEpochMilliseconds(currentMillis)

            folderDao.updateName(id = "1", userId = testUserId, name = "New Name", updatedAt = updateTime)

            val updated = folderDao.getFolderByIdAndUser("1", testUserId)
            assertNotNull(updated)
            assertEquals("New Name", updated?.name)
            assertTrue(updated?.isSynced == false)

            assertEquals(updateTime.toEpochMilliseconds(), updated?.updatedAt?.toEpochMilliseconds())
        }

    @Test
    fun getUnsyncedFolders_returnsOnlyFoldersWithIsSyncedFalse() =
        runTest {
            val unsynced = createTestFolder(id = "1", userId = testUserId, isSynced = false)
            val synced = createTestFolder(id = "2", userId = testUserId, isSynced = true)

            folderDao.insert(unsynced)
            folderDao.insert(synced)

            val result = folderDao.getUnsyncedFolders(testUserId)

            assertEquals(1, result.size)
            assertEquals("1", result[0].id)
        }

    @Test
    fun hardDeleteById_physicallyRemovesRowFromDatabase() =
        runTest {
            val folder = createTestFolder(id = "1", userId = testUserId)
            folderDao.insert(folder)

            assertNotNull(folderDao.getFolderByIdAndUser("1", testUserId))

            folderDao.hardDeleteById("1", testUserId)

            assertTrue(folderDao.getUnsyncedFolders(testUserId).isEmpty())
        }

    private fun createTestFolder(
        id: String,
        userId: String,
        name: String = "Folder_$id",
        isSynced: Boolean = false,
        isDeleted: Boolean = false,
    ): FolderEntity =
        FolderEntity(
            id = id,
            userId = userId,
            name = name,
            createdAt = Clock.System.now(),
            updatedAt = Clock.System.now(),
            isSynced = isSynced,
            isDeleted = isDeleted,
            metadata = emptyMap(),
        )
}

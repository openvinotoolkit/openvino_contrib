package com.itlab.data.repository

import com.itlab.data.dao.FolderDao
import com.itlab.data.dao.MediaDao
import com.itlab.data.dao.NoteDao
import com.itlab.data.mapper.NoteFolderMapper
import com.itlab.domain.model.NoteFolder
import io.mockk.Runs
import io.mockk.coEvery
import io.mockk.coVerify
import io.mockk.just
import io.mockk.mockk
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.flowOf
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.time.Clock

class NoteFolderRepositoryImplTest {
    private val folderDao = mockk<FolderDao>(relaxed = true)
    private val noteDao = mockk<NoteDao>(relaxed = true)
    private val mediaDao = mockk<MediaDao>(relaxed = true)
    private val mapper = NoteFolderMapper()
    private val repository = NoteFolderRepositoryImpl(folderDao, noteDao, mediaDao, mapper)

    private val testUserId = "test_user_1"
    private val testFolder =
        NoteFolder(
            id = "1",
            name = "Work",
            createdAt = Clock.System.now(),
            updatedAt = Clock.System.now(),
            metadata = emptyMap(),
            userId = testUserId,
        )

    @Test
    fun `observeFolders emits mapped domain list`() =
        runTest {
            val entity = mapper.toEntity(testFolder)
            coEvery { folderDao.getActiveFoldersByUserId(testUserId) } returns flowOf(listOf(entity))

            val result = repository.observeFolders(testUserId).first()

            assertEquals(1, result.size)
            assertEquals(testFolder.name, result[0].name)
        }

    @Test
    fun `renameFolder calls dao updateName with current timestamp`() =
        runTest {
            coEvery { folderDao.updateName(any(), any(), any(), any()) } just Runs

            repository.renameFolder("1", testUserId, "New Name")

            coVerify(exactly = 1) {
                folderDao.updateName(id = "1", userId = testUserId, name = "New Name", updatedAt = any())
            }
        }

    @Test
    fun `deleteFolder soft-deletes notes in folder then folder`() =
        runTest {
            coEvery { noteDao.getNotesByFolderAndUser("1", testUserId) } returns flowOf(emptyList())
            coEvery { noteDao.softDeleteByFolderId(any(), any(), any()) } just Runs
            coEvery { folderDao.softDeleteById(any(), any(), any()) } just Runs

            repository.deleteFolder("1", testUserId)

            coVerify(exactly = 1) {
                noteDao.softDeleteByFolderId(folderId = "1", userId = testUserId, timestamp = any())
            }
            coVerify(exactly = 1) {
                folderDao.softDeleteById(id = "1", userId = testUserId, updatedAt = any())
            }
        }

    @Test
    fun `getFolderById returns mapped folder when found`() =
        runTest {
            val entity = mapper.toEntity(testFolder)
            coEvery { folderDao.getFolderByIdAndUser("1", testUserId) } returns entity

            val result = repository.getFolderById("1", testUserId)

            assertEquals("Work", result?.name)
            assertEquals(testUserId, result?.userId)
        }

    @Test
    fun `getFolderById returns null when dao returns null`() =
        runTest {
            coEvery { folderDao.getFolderByIdAndUser("any", testUserId) } returns null

            val result = repository.getFolderById("any", testUserId)

            assertNull(result)
        }

    @Test
    fun `observeFolders emits empty list when dao is empty`() =
        runTest {
            coEvery { folderDao.getActiveFoldersByUserId(testUserId) } returns flowOf(emptyList())

            val result = repository.observeFolders(testUserId).first()

            assertTrue(result.isEmpty())
        }

    @Test
    fun `updateFolder calls dao update with reset sync flag`() =
        runTest {
            coEvery { folderDao.update(any()) } just Runs

            repository.updateFolder(testFolder)

            coVerify(exactly = 1) {
                folderDao.update(match { it.id == "1" && !it.isSynced })
            }
        }

    @Test
    fun `createFolder inserts entity with reset sync flags and returns correct id`() =
        runTest {
            val folder = NoteFolder(userId = testUserId, id = "folder_777", name = "Test Folder")
            coEvery { folderDao.insert(any()) } just Runs

            val resultId = repository.createFolder(folder)

            coVerify(exactly = 1) {
                folderDao.insert(
                    match {
                        it.id == "folder_777" && !it.isSynced && !it.isDeleted
                    },
                )
            }
            assertEquals("folder_777", resultId)
        }
}

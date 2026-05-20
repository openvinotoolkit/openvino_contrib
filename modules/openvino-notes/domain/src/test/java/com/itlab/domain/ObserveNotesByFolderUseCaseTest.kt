package com.itlab.domain

import com.itlab.domain.model.Note
import com.itlab.domain.repository.NotesRepository
import com.itlab.domain.usecase.noteusecase.GetUserIdUseCase
import com.itlab.domain.usecase.noteusecase.ObserveNotesByFolderUseCase
import io.mockk.MockKAnnotations
import io.mockk.coEvery
import io.mockk.every
import io.mockk.impl.annotations.MockK
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.flowOf
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Before
import org.junit.Test
import kotlin.time.Clock
import kotlin.time.Clock.System.now
import kotlin.time.Instant

class ObserveNotesByFolderUseCaseTest {
    @MockK
    lateinit var repo: NotesRepository

    private lateinit var observeNotesByFolderUseCase: ObserveNotesByFolderUseCase

    @MockK
    lateinit var getUserIdUsecase: GetUserIdUseCase

    private val testUserId = "test_user_1"

    @Before
    fun setUp() {
        MockKAnnotations.init(this)
        every { getUserIdUsecase() } returns testUserId
        observeNotesByFolderUseCase = ObserveNotesByFolderUseCase(repo, getUserIdUsecase)
    }

    @Test
    fun `invoke should return all notes when folderId is null`() =
        runBlocking {
            val now = Instant.fromEpochMilliseconds(System.currentTimeMillis())
            val notes =
                listOf(
                    Note(userId = "u1", id = "1", folderId = "folder_1", createdAt = now, updatedAt = now),
                    Note(userId = "u1", id = "2", folderId = "folder_2", createdAt = now, updatedAt = now),
                )
            coEvery { repo.observeNotes(testUserId) } returns flowOf(notes)

            val result = observeNotesByFolderUseCase(null).first()

            assertEquals(2, result.size)
        }

    @Test
    fun `invoke should filter notes by folderId when it is not null`() =
        runBlocking {
            val targetFolder = "folder_1"
            val now = Instant.fromEpochMilliseconds(System.currentTimeMillis())
            val notes =
                listOf(
                    Note(userId = "u1", id = "1", folderId = targetFolder, createdAt = now, updatedAt = now),
                    Note(userId = "u1", id = "2", folderId = "other_folder", createdAt = now, updatedAt = now),
                )
            coEvery { repo.observeNotes(testUserId) } returns flowOf(notes)

            val result = observeNotesByFolderUseCase(targetFolder).first()

            assertEquals(1, result.size)
            assertEquals(targetFolder, result[0].folderId)
        }

    @Test
    fun `verify flow mapping is executed`() =
        runBlocking {
            val notes =
                listOf(
                    Note(
                        userId = "u1",
                        id = "1",
                        folderId = "A",
                        createdAt = Clock.System.now(),
                        updatedAt = Clock.System.now(),
                    ),
                )
            coEvery { repo.observeNotes(testUserId) } returns flowOf(notes)

            val result = observeNotesByFolderUseCase("A").first()

            assertEquals(1, result.size)
        }
}

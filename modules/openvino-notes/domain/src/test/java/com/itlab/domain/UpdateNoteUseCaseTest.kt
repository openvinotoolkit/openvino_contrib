package com.itlab.domain

import com.itlab.domain.model.Note
import com.itlab.domain.repository.NotesRepository
import com.itlab.domain.usecase.noteusecase.GetUserIdUseCase
import com.itlab.domain.usecase.noteusecase.UpdateNoteUseCase
import io.mockk.MockKAnnotations
import io.mockk.coEvery
import io.mockk.coVerify
import io.mockk.every
import io.mockk.impl.annotations.MockK
import kotlinx.coroutines.flow.flowOf
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import kotlin.time.Instant

class UpdateNoteUseCaseTest {
    @MockK
    lateinit var repo: NotesRepository

    private lateinit var updateNoteUseCase: UpdateNoteUseCase

    @MockK
    lateinit var getUserIdUsecase: GetUserIdUseCase

    private val testUserId = "test_user_1"

    @Before
    fun setUp() {
        MockKAnnotations.init(this)
        every { getUserIdUsecase() } returns testUserId
        updateNoteUseCase = UpdateNoteUseCase(repo, getUserIdUsecase)
    }

    @Test
    fun `invoke should return failure when duplicate title exists in same folder`() =
        runBlocking {
            val now = Instant.fromEpochMilliseconds(System.currentTimeMillis())
            val folderId = "folder_1"

            val existingNote =
                Note(
                    userId = testUserId,
                    id = "note_1",
                    title = "Meeting",
                    folderId = folderId,
                    createdAt = now,
                    updatedAt = now,
                )
            val noteToUpdate =
                Note(
                    userId = testUserId,
                    id = "note_2",
                    title = "  meeting  ",
                    folderId = folderId,
                    createdAt = now,
                    updatedAt = now,
                )

            coEvery { repo.observeNotes(testUserId) } returns flowOf(listOf(existingNote))

            val result = updateNoteUseCase(noteToUpdate)

            assertTrue(result.isFailure)
            assertEquals("Note with title 'meeting' already exists in this folder", result.exceptionOrNull()?.message)
            coVerify(exactly = 0) { repo.updateNote(any()) }
        }

    @Test
    fun `invoke should succeed when updating same note with same title`() =
        runBlocking {
            val now = Instant.fromEpochMilliseconds(System.currentTimeMillis())
            val note =
                Note(testUserId, id = "1", title = "Original", folderId = "A", createdAt = now, updatedAt = now)

            coEvery { repo.observeNotes(testUserId) } returns flowOf(listOf(note))
            coEvery { repo.updateNote(any()) } returns Unit

            val result = updateNoteUseCase(note)

            assertTrue(result.isSuccess)
            coVerify { repo.updateNote(any()) }
        }

    @Test
    fun `invoke should succeed when same title is in different folder`() =
        runBlocking {
            val now = Instant.fromEpochMilliseconds(System.currentTimeMillis())
            val existingNote =
                Note(testUserId, id = "1", title = "Same", folderId = "Folder_A", createdAt = now, updatedAt = now)
            val noteToUpdate =
                Note(testUserId, id = "2", title = "Same", folderId = "Folder_B", createdAt = now, updatedAt = now)

            coEvery { repo.observeNotes(testUserId) } returns flowOf(listOf(existingNote))
            coEvery { repo.updateNote(any()) } returns Unit

            val result = updateNoteUseCase(noteToUpdate)

            assertTrue(result.isSuccess)
        }
}

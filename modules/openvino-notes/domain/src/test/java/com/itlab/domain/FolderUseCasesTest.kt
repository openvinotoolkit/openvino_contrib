package com.itlab.domain

import com.itlab.domain.model.Note
import com.itlab.domain.model.NoteFolder
import com.itlab.domain.repository.NoteFolderRepository
import com.itlab.domain.repository.NotesRepository
import com.itlab.domain.usecase.folderusecase.CreateFolderUseCase
import com.itlab.domain.usecase.folderusecase.DeleteFolderUseCase
import com.itlab.domain.usecase.folderusecase.GetFolderUseCase
import com.itlab.domain.usecase.folderusecase.ObserveFoldersUseCase
import com.itlab.domain.usecase.folderusecase.UpdateFolderUseCase
import com.itlab.domain.usecase.noteusecase.GetUserIdUseCase
import io.mockk.MockKAnnotations
import io.mockk.every
import io.mockk.impl.annotations.MockK
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Before
import org.junit.Test

class FolderUseCasesTest {
    @MockK
    lateinit var getUserIdUsecase: GetUserIdUseCase

    private val testUserId = "user_folder_test"

    private class FakeFolderRepo : NoteFolderRepository {
        private val store = mutableMapOf<String, NoteFolder>()
        private val flow = MutableStateFlow<List<NoteFolder>>(emptyList())

        override fun observeFolders(userId: String) = flow

        override suspend fun createFolder(folder: NoteFolder): String {
            store[folder.id] = folder
            flow.value = store.values.toList()
            return folder.id
        }

        override suspend fun renameFolder(
            id: String,
            userId: String,
            name: String,
        ) {
            val folder = store[id] ?: return
            val updated = folder.copy(name = name)
            store[id] = updated
            flow.value = store.values.toList()
        }

        override suspend fun deleteFolder(
            id: String,
            userId: String,
        ) {
            store.remove(id)
            flow.value = store.values.toList()
        }

        override suspend fun getFolderById(
            id: String,
            userId: String,
        ): NoteFolder? = store[id]

        override suspend fun updateFolder(folder: NoteFolder) {
            store[folder.id] = folder
            flow.value = store.values.toList()
        }
    }

    private class FakeNotesRepo : NotesRepository {
        override fun observeNotes(userId: String) = MutableStateFlow<List<Note>>(emptyList())

        override fun observeNotesByFolder(
            folderId: String,
            userId: String,
        ) = MutableStateFlow<List<Note>>(emptyList())

        override suspend fun getNoteById(
            id: String,
            userId: String,
        ): Note? = null

        override suspend fun createNote(note: Note): String = note.id

        override suspend fun updateNote(note: Note) = Unit

        override suspend fun deleteNote(
            id: String,
            userId: String,
        ) = Unit
    }

    @Before
    fun setUp() {
        MockKAnnotations.init(this)

        every { getUserIdUsecase() } returns testUserId
    }

    @Test
    fun createFolder_and_getFolder() =
        runBlocking {
            val repo = FakeFolderRepo()

            val create = CreateFolderUseCase(repo, getUserIdUsecase)
            val get = GetFolderUseCase(repo, getUserIdUsecase)

            val folder = NoteFolder(testUserId, name = "Test")

            val id = create(folder).getOrThrow()

            val result = get(id)

            assertEquals("Test", result?.name)
        }

    @Test
    fun updateFolder_works() =
        runBlocking {
            val repo = FakeFolderRepo()

            val create = CreateFolderUseCase(repo, getUserIdUsecase)
            val update = UpdateFolderUseCase(repo, getUserIdUsecase)
            val get = GetFolderUseCase(repo, getUserIdUsecase)

            val folder = NoteFolder(testUserId, name = "Old")

            val id = create(folder).getOrThrow()

            val created = get(id)!!

            val updated = created.copy(name = "New")
            update(updated).getOrThrow()
            val result = get(id)

            assertEquals("New", result?.name)
        }

    @Test
    fun deleteFolder_works() =
        runBlocking {
            val repo = FakeFolderRepo()

            val create = CreateFolderUseCase(repo, getUserIdUsecase)
            val delete = DeleteFolderUseCase(repo, FakeNotesRepo(), getUserIdUsecase)
            val get = GetFolderUseCase(repo, getUserIdUsecase)

            val folder = NoteFolder(testUserId, name = "Test")

            val id = create(folder).getOrThrow()

            delete(id).getOrThrow()

            val result = get(id)

            assertNull(result)
        }

    @Test
    fun observeFolders_emitsData() =
        runBlocking {
            val repo = FakeFolderRepo()
            val create = CreateFolderUseCase(repo, getUserIdUsecase)
            val observe = ObserveFoldersUseCase(repo, getUserIdUsecase)

            create(NoteFolder(testUserId, name = "A")).getOrThrow()

            val list = observe().first()

            assertEquals(1, list.size)
        }

    @Test
    fun createFolder_blankName_returnsFailure() =
        runBlocking {
            val repo = FakeFolderRepo()
            val create = CreateFolderUseCase(repo, getUserIdUsecase)
            val result = create(NoteFolder(testUserId, name = "   "))
            assertEquals(true, result.isFailure)
            assertEquals("Folder name must not be blank", result.exceptionOrNull()?.message)
        }
}

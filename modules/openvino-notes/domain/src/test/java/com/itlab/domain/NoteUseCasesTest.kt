package com.itlab.domain

import com.itlab.domain.model.ContentItem
import com.itlab.domain.model.Note
import com.itlab.domain.model.NoteFolder
import com.itlab.domain.repository.NoteFolderRepository
import com.itlab.domain.repository.NotesRepository
import com.itlab.domain.usecase.noteusecase.AddTagUseCase
import com.itlab.domain.usecase.noteusecase.CreateNoteUseCase
import com.itlab.domain.usecase.noteusecase.DeleteNoteUseCase
import com.itlab.domain.usecase.noteusecase.DeleteTagUseCase
import com.itlab.domain.usecase.noteusecase.DuplicateNoteUseCase
import com.itlab.domain.usecase.noteusecase.GetAllFavoritesUseCase
import com.itlab.domain.usecase.noteusecase.GetNoteUseCase
import com.itlab.domain.usecase.noteusecase.GetNotesByTagUseCase
import com.itlab.domain.usecase.noteusecase.GetUserIdUseCase
import com.itlab.domain.usecase.noteusecase.MoveNoteToFolderUseCase
import com.itlab.domain.usecase.noteusecase.ObserveNotesUseCase
import com.itlab.domain.usecase.noteusecase.SearchNotesUseCase
import com.itlab.domain.usecase.noteusecase.SwitchFavoriteUseCase
import com.itlab.domain.usecase.noteusecase.UpdateNoteUseCase
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

class NoteUseCasesTest {
    private val testUserId = "test_user_1"

    @MockK
    lateinit var getUserIdUsecase: GetUserIdUseCase

    private class FakeNotesRepo : NotesRepository {
        private val store = mutableMapOf<String, Note>()
        private val flow = MutableStateFlow<List<Note>>(emptyList())

        override fun observeNotes(userId: String) = flow

        override fun observeNotesByFolder(
            folderId: String,
            userId: String,
        ) = flow

        override suspend fun getNoteById(
            id: String,
            userId: String,
        ): Note? = store[id]

        override suspend fun createNote(note: Note): String {
            store[note.id] = note
            flow.value = store.values.toList()
            return note.id
        }

        override suspend fun updateNote(note: Note) {
            store[note.id] = note
            flow.value = store.values.toList()
        }

        override suspend fun deleteNote(
            id: String,
            userId: String,
        ) {
            store.remove(id)
            flow.value = store.values.toList()
        }
    }

    private class FakeFolderRepo : NoteFolderRepository {
        private val store = mutableMapOf<String, NoteFolder>()

        override fun observeFolders(userId: String) = MutableStateFlow(emptyList<NoteFolder>())

        override suspend fun createFolder(folder: NoteFolder): String {
            store[folder.id] = folder
            return folder.id
        }

        override suspend fun renameFolder(
            id: String,
            userId: String,
            name: String,
        ) = Unit

        override suspend fun deleteFolder(
            id: String,
            userId: String,
        ) = Unit

        override suspend fun getFolderById(
            id: String,
            userId: String,
        ): NoteFolder? = store[id]

        override suspend fun updateFolder(folder: NoteFolder) = Unit
    }

    @Before
    fun setUp() {
        MockKAnnotations.init(this)
        every { getUserIdUsecase() } returns testUserId
    }

    @Test
    fun create_update_delete_note() =
        runBlocking {
            val repo = FakeNotesRepo()

            val create = CreateNoteUseCase(repo, getUserIdUsecase)
            val update = UpdateNoteUseCase(repo, getUserIdUsecase)
            val delete = DeleteNoteUseCase(repo, getUserIdUsecase)
            val get = GetNoteUseCase(repo, getUserIdUsecase)

            val note = Note(id = "n1", title = "A", userId = testUserId)

            val id = create(note).getOrThrow()

            val created = get(id)!!

            val updated = created.copy(title = "B")
            update(updated).getOrThrow()

            val result = get(id)
            assertEquals("B", result?.title)

            delete(id)

            val result2 = get(id)
            assertNull(result2)
        }

    @Test
    fun moveNoteToFolder_works() =
        runBlocking {
            val notesRepo = FakeNotesRepo()
            val folderRepo = FakeFolderRepo()

            val move = MoveNoteToFolderUseCase(notesRepo, folderRepo, getUserIdUsecase)
            val createNote = CreateNoteUseCase(notesRepo, getUserIdUsecase)

            val folder = NoteFolder(testUserId, id = "f1", name = "Folder")
            folderRepo.createFolder(folder)

            val note = Note(id = "n1", title = "Note", userId = testUserId)

            val noteId = createNote(note).getOrThrow()

            move("f1", noteId).getOrThrow()
            val updated = notesRepo.getNoteById(noteId, testUserId)

            assertEquals("f1", updated?.folderId)
        }

    @Test
    fun observeNotes_returnsData() =
        runBlocking {
            val repo = FakeNotesRepo()
            val observe = ObserveNotesUseCase(repo, getUserIdUsecase)
            val create = CreateNoteUseCase(repo, getUserIdUsecase)

            create(Note(id = "n1", title = "Test", userId = testUserId)).getOrThrow()

            val list = observe().first()

            assertEquals(1, list.size)
        }

    @Test
    fun addTag_addsTagToNote() =
        runBlocking {
            val repo = FakeNotesRepo()
            val useCase = AddTagUseCase(repo, getUserIdUsecase)

            val note =
                Note(
                    id = "n1",
                    title = "Test",
                    tags = setOf("old"),
                    userId = testUserId,
                )
            repo.createNote(note)

            useCase("n1", "new-tag").getOrThrow()

            val updated = repo.getNoteById("n1", testUserId)

            assertEquals(setOf("old", "new-tag"), updated?.tags)
        }

    @Test
    fun addTag_throwsIfNoteNotFound() =
        runBlocking {
            val repo = FakeNotesRepo()
            val useCase = AddTagUseCase(repo, getUserIdUsecase)

            val result = useCase("missing_id", "tag")
            assertEquals(true, result.isFailure)
            assertEquals("Note not found: missing_id", result.exceptionOrNull()?.message)
        }

    @Test
    fun deleteTag_removesTagFromNote() =
        runBlocking {
            val repo = FakeNotesRepo()
            val useCase = DeleteTagUseCase(repo, getUserIdUsecase)

            val note =
                Note(
                    id = "n2",
                    title = "Test",
                    tags = setOf("old", "remove-me"),
                    userId = testUserId,
                )
            repo.createNote(note)

            useCase("n2", "remove-me").getOrThrow()

            val updated = repo.getNoteById("n2", testUserId)

            assertEquals(setOf("old"), updated?.tags)
        }

    @Test
    fun deleteTag_throwsIfNoteNotFound() =
        runBlocking {
            val repo = FakeNotesRepo()
            val useCase = DeleteTagUseCase(repo, getUserIdUsecase)

            val result = useCase("missing_id", "tag")
            assertEquals(true, result.isFailure)
            assertEquals("Note not found: missing_id", result.exceptionOrNull()?.message)
        }

    @Test
    fun duplicateNote_createsCopyWithNewIdAndCopiedTitle() =
        runBlocking {
            val repo = FakeNotesRepo()
            val useCase = DuplicateNoteUseCase(repo, getUserIdUsecase)

            val original =
                Note(
                    id = "n3",
                    title = "Hello",
                    tags = setOf("kotlin"),
                    isFavorite = true,
                    summary = "summary",
                    userId = testUserId,
                )
            repo.createNote(original)

            val newId = useCase("n3").getOrThrow()

            val duplicated = repo.getNoteById(newId, testUserId)

            assertEquals(true, duplicated != null)
            assertEquals("Hello Copy", duplicated?.title)
            assertEquals(setOf("kotlin"), duplicated?.tags)
            assertEquals(true, duplicated?.isFavorite)
            assertEquals("summary", duplicated?.summary)
            assertEquals("n3", original.id)
            assertEquals(false, original.id == newId)
        }

    @Test
    fun duplicateNote_usesCopyWhenTitleBlank() =
        runBlocking {
            val repo = FakeNotesRepo()
            val useCase = DuplicateNoteUseCase(repo, getUserIdUsecase)

            val original =
                Note(
                    id = "n4",
                    title = "   ",
                    userId = testUserId,
                )
            repo.createNote(original)

            val newId = useCase("n4").getOrThrow()

            val duplicated = repo.getNoteById(newId, testUserId)

            assertEquals("Copy", duplicated?.title)
        }

    @Test
    fun duplicateNote_throwsIfNoteNotFound() =
        runBlocking {
            val repo = FakeNotesRepo()
            val useCase = DuplicateNoteUseCase(repo, getUserIdUsecase)

            val result = useCase("missing_id")
            assertEquals(true, result.isFailure)
            assertEquals("Note not found: missing_id", result.exceptionOrNull()?.message)
        }

    @Test
    fun getAllFavorites_returnsOnlyFavoriteNotes() =
        runBlocking {
            val repo = FakeNotesRepo()
            val useCase = GetAllFavoritesUseCase(repo, getUserIdUsecase)

            repo.createNote(
                Note(
                    id = "n5",
                    title = "A",
                    isFavorite = true,
                    userId = testUserId,
                ),
            )
            repo.createNote(
                Note(
                    id = "n6",
                    title = "B",
                    isFavorite = false,
                    userId = testUserId,
                ),
            )

            val list = useCase().first()

            assertEquals(1, list.size)
            assertEquals("n5", list.first().id)
        }

    @Test
    fun switchFavorite_turnsFavoriteOnAndOff() =
        runBlocking {
            val repo = FakeNotesRepo()
            val useCase = SwitchFavoriteUseCase(repo, getUserIdUsecase)

            val note =
                Note(
                    id = "n7",
                    title = "Fav",
                    isFavorite = false,
                    userId = testUserId,
                )
            repo.createNote(note)

            useCase("n7").getOrThrow()
            val afterFirstSwitch = repo.getNoteById("n7", testUserId)
            assertEquals(true, afterFirstSwitch?.isFavorite)

            useCase("n7").getOrThrow()
            val afterSecondSwitch = repo.getNoteById("n7", testUserId)
            assertEquals(false, afterSecondSwitch?.isFavorite)
        }

    @Test
    fun switchFavorite_throwsIfNoteNotFound() =
        runBlocking {
            val repo = FakeNotesRepo()
            val useCase = SwitchFavoriteUseCase(repo, getUserIdUsecase)

            val result = useCase("missing_id")
            assertEquals(true, result.isFailure)
            assertEquals("Note not found", result.exceptionOrNull()?.message)
        }

    @Test
    fun getNotesByTag_returnsOnlyMatchingTag() =
        runBlocking {
            val repo = FakeNotesRepo()
            val useCase = GetNotesByTagUseCase(repo, getUserIdUsecase)

            repo.createNote(Note(id = "n10", tags = setOf("work", "urgent"), userId = testUserId))
            repo.createNote(Note(id = "n11", tags = setOf("personal"), userId = testUserId))

            val result = useCase("URGENT").first()

            assertEquals(1, result.size)
            assertEquals("n10", result.first().id)
        }

    @Test
    fun searchNotes_findsByTitleAndTextContent() =
        runBlocking {
            val repo = FakeNotesRepo()
            val useCase = SearchNotesUseCase(repo, getUserIdUsecase)

            repo.createNote(
                Note(
                    id = "n8",
                    title = "Планы на отпуск",
                    userId = testUserId,
                ),
            )
            repo.createNote(
                Note(
                    id = "n9",
                    title = "Покупки",
                    contentItems = listOf(ContentItem.Text(text = "Купить молоко и хлеб")),
                    userId = testUserId,
                ),
            )

            val result = useCase("молоко").first()

            assertEquals(1, result.size)
            assertEquals("n9", result.first().id)
        }

    @Test
    fun addTag_trimsIncomingTag() =
        runBlocking {
            val repo = FakeNotesRepo()
            val useCase = AddTagUseCase(repo, getUserIdUsecase)
            repo.createNote(Note(id = "n1", title = "Test", tags = emptySet(), userId = testUserId))

            useCase("n1", "  kotlin  ")

            val updated = repo.getNoteById("n1", testUserId)
            assertEquals(setOf("kotlin"), updated?.tags)
        }
}

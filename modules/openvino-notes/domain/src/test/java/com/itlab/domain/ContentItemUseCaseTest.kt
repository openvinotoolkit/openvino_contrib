package com.itlab.domain

import com.itlab.domain.model.ContentItem
import com.itlab.domain.model.Note
import com.itlab.domain.repository.NotesRepository
import com.itlab.domain.usecase.contentusecase.AddContentItemUseCase
import com.itlab.domain.usecase.contentusecase.CreateContentItemUseCase
import com.itlab.domain.usecase.contentusecase.DeleteContentItemUseCase
import com.itlab.domain.usecase.contentusecase.GetContentItemUseCase
import com.itlab.domain.usecase.noteusecase.GetUserIdUseCase
import io.mockk.MockKAnnotations
import io.mockk.every
import io.mockk.impl.annotations.MockK
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Before
import org.junit.Test

class ContentItemUseCaseTest {
    val testUserId = "testID"

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

    @Before
    fun setUp() {
        MockKAnnotations.init(this)
        every { getUserIdUsecase() } returns testUserId
    }

    @Test
    fun add_get_delete_content_item_flow() =
        runBlocking {
            val repo = FakeNotesRepo()

            val createItem = CreateContentItemUseCase()
            val addItem = AddContentItemUseCase(repo, getUserIdUsecase)
            val getItem = GetContentItemUseCase(repo, getUserIdUsecase)
            val deleteItem = DeleteContentItemUseCase(repo, getUserIdUsecase)

            val note = Note(id = "n1", title = "Test", userId = testUserId)
            repo.createNote(note)

            val item =
                createItem(
                    ContentItem.Text(
                        text = "Hello",
                    ),
                )

            addItem("n1", item).getOrThrow()

            val found = getItem("n1", item.id).getOrThrow()
            assertEquals("Hello", (found as ContentItem.Text).text)

            deleteItem("n1", item.id).getOrThrow()

            val afterDelete = getItem("n1", item.id).getOrThrow()
            assertNull(afterDelete)
        }

    @Test
    fun addContentItem_without_id_returnsFailure() =
        runBlocking {
            val repo = FakeNotesRepo()
            val addItem = AddContentItemUseCase(repo, getUserIdUsecase)

            val note = Note(id = "n1", title = "Test", userId = testUserId)
            repo.createNote(note)

            val badItem =
                ContentItem.Text(
                    id = "",
                    text = "Hello",
                )

            val result = addItem("n1", badItem)
            assertEquals(true, result.isFailure)
            assertEquals("ContentItem id must not be blank", result.exceptionOrNull()?.message)
        }

    @Test
    fun add_multiple_content_items() =
        runBlocking {
            val repo = FakeNotesRepo()
            val createItem = CreateContentItemUseCase()
            val addItem = AddContentItemUseCase(repo, getUserIdUsecase)
            val getItem = GetContentItemUseCase(repo, getUserIdUsecase)

            val note = Note(id = "n1", title = "Test", userId = testUserId)
            repo.createNote(note)

            val item1 = createItem(ContentItem.Text(text = "A"))
            val item2 = createItem(ContentItem.Text(text = "B"))

            addItem("n1", item1).getOrThrow()
            addItem("n1", item2).getOrThrow()

            val found1 = getItem("n1", item1.id).getOrThrow()
            val found2 = getItem("n1", item2.id).getOrThrow()

            assertEquals("A", (found1 as ContentItem.Text).text)
            assertEquals("B", (found2 as ContentItem.Text).text)
        }

    @Test
    fun delete_non_existing_item_does_nothing() =
        runBlocking {
            val repo = FakeNotesRepo()
            val deleteItem = DeleteContentItemUseCase(repo, getUserIdUsecase)

            val note = Note(id = "n1", title = "Test", userId = testUserId)
            repo.createNote(note)

            deleteItem("n1", "wrong-id").getOrThrow()

            val result = repo.getNoteById("n1", testUserId)
            assertEquals(0, result?.contentItems?.size)
        }

    @Test
    fun content_item_does_not_affect_original_note_instance() =
        runBlocking {
            val repo = FakeNotesRepo()
            val createItem = CreateContentItemUseCase()
            val addItem = AddContentItemUseCase(repo, getUserIdUsecase)

            val note = Note(id = "n1", title = "Test", userId = testUserId)
            repo.createNote(note)

            val item = createItem(ContentItem.Text(text = "Hello"))

            addItem("n1", item).getOrThrow()

            assertEquals(0, note.contentItems.size)
        }

    @Test
    fun add_duplicate_content_item_id_throws() =
        runBlocking {
            val repo = FakeNotesRepo()
            val addItem = AddContentItemUseCase(repo, getUserIdUsecase)
            val note = Note(id = "n1", title = "Test", userId = testUserId)
            repo.createNote(note)

            val item = ContentItem.Text(id = "fixed-id", text = "Hello")

            addItem("n1", item).getOrThrow()

            val result = addItem("n1", item.copy(text = "World"))
            assertEquals(true, result.isFailure)
            assertEquals(
                "Content item with id 'fixed-id' already exists in note 'n1'",
                result.exceptionOrNull()?.message,
            )
        }
}

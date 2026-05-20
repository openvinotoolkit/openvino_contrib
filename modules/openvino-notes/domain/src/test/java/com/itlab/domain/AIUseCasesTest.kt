package com.itlab.domain

import com.itlab.domain.ai.NoteAiService
import com.itlab.domain.model.ContentItem
import com.itlab.domain.model.DataSource
import com.itlab.domain.model.Note
import com.itlab.domain.repository.NotesRepository
import com.itlab.domain.usecase.aiusecase.SuggestSummaryUseCase
import com.itlab.domain.usecase.aiusecase.SuggestTagsUseCase
import com.itlab.domain.usecase.noteusecase.ApplySummaryUseCase
import com.itlab.domain.usecase.noteusecase.ApplyTagsUseCase
import com.itlab.domain.usecase.noteusecase.GetUserIdUseCase
import io.mockk.MockKAnnotations
import io.mockk.every
import io.mockk.impl.annotations.MockK
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Before
import org.junit.Test

class AIUseCasesTest {
    private val testUserId = "test_user_1"

    @MockK
    lateinit var getUserIdUsecase: GetUserIdUseCase

    private class FakeNotesRepo : NotesRepository {
        private val store = mutableMapOf<String, Note>()
        private val flow = MutableStateFlow<List<Note>>(emptyList())

        override fun observeNotes(userId: String): Flow<List<Note>> = flow

        override fun observeNotesByFolder(
            folderId: String,
            userId: String,
        ): Flow<List<Note>> = flow.map { notes -> notes.filter { it.folderId == folderId } }

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

    private class FakeNoteAiService : NoteAiService {
        var summaryInput: String? = null
        var textTagsInput: String? = null
        var imageTagsInput: List<String> = emptyList()

        var summaryResult: String = "AI summary"
        var textTagsResult: Set<String> = setOf("text-tag-1", "text-tag-2")
        var imageTagsResult: Set<String> = setOf("image-tag-1", "image-tag-2")

        override suspend fun summarize(text: String): String {
            summaryInput = text
            return summaryResult
        }

        override suspend fun tagIMGs(img: List<String>): Set<String> {
            imageTagsInput = img
            return imageTagsResult
        }

        override suspend fun tagTXT(text: String): Set<String> {
            textTagsInput = text
            return textTagsResult
        }
    }

    @Before
    fun setUp() {
        MockKAnnotations.init(this)
        every { getUserIdUsecase() } returns testUserId
    }

    @Test
    fun suggestSummary_returnsDataAndSendsJoinedTextToAi() =
        runBlocking {
            val repo = FakeNotesRepo()
            val ai = FakeNoteAiService()
            val useCase = SuggestSummaryUseCase(ai, repo, getUserIdUsecase)

            val note =
                Note(
                    id = "n1",
                    title = "Test",
                    userId = testUserId,
                    contentItems =
                        listOf(
                            ContentItem.Text(text = "Hello"),
                            ContentItem.Image(
                                source = DataSource(localPath = "path/to/img"),
                                mimeType = "image/png",
                            ),
                        ),
                )

            repo.createNote(note)

            val result = useCase("n1")

            assertEquals("AI summary", result.getOrThrow())
            assertEquals("Hello", ai.summaryInput)
        }

    @Test
    fun suggestSummary_throwsIfNoteNotFound() =
        runBlocking {
            val repo = FakeNotesRepo()
            val ai = FakeNoteAiService()
            val useCase = SuggestSummaryUseCase(ai, repo, getUserIdUsecase)

            val result = useCase("missing_id")
            assertEquals(true, result.isFailure)
            assertEquals("Note not found: missing_id", result.exceptionOrNull()?.message)
        }

    @Test
    fun suggestTags_returnsMergedTags_andSendsTextAndImagesToAi() =
        runBlocking {
            val repo = FakeNotesRepo()
            val ai = FakeNoteAiService()
            val useCase = SuggestTagsUseCase(ai, repo, getUserIdUsecase)

            val note =
                Note(
                    id = "n2",
                    title = "Tags",
                    userId = testUserId,
                    contentItems =
                        listOf(
                            ContentItem.Text(text = "First line"),
                            ContentItem.Text(text = "Second line"),
                            ContentItem.Image(
                                source = DataSource(localPath = "/local/image.png"),
                                mimeType = "image/png",
                            ),
                            ContentItem.Image(
                                source = DataSource(remoteUrl = "https://example.com/image.jpg"),
                                mimeType = "image/jpg",
                            ),
                            ContentItem.Link(url = "https://kotlinlang.org"),
                        ),
                )

            repo.createNote(note)

            val result = useCase("n2")

            assertEquals("First line\nSecond line", ai.textTagsInput)

            assertEquals(
                listOf("/local/image.png", "https://example.com/image.jpg"),
                ai.imageTagsInput,
            )

            assertEquals(
                setOf("text-tag-1", "text-tag-2", "image-tag-1", "image-tag-2"),
                result.getOrThrow(),
            )
        }

    @Test
    fun suggestTags_throwsIfNoteNotFound() =
        runBlocking {
            val repo = FakeNotesRepo()
            val ai = FakeNoteAiService()
            val useCase = SuggestTagsUseCase(ai, repo, getUserIdUsecase)

            val result = useCase("missing_id")
            assertEquals(true, result.isFailure)
            assertEquals("Note not found: missing_id", result.exceptionOrNull()?.message)
        }

    @Test
    fun applySummary_updatesNoteSummary() =
        runBlocking {
            val repo = FakeNotesRepo()
            val useCase = ApplySummaryUseCase(repo, getUserIdUsecase)

            val note =
                Note(
                    id = "n3",
                    title = "Summary",
                    summary = "old summary",
                    userId = testUserId,
                )

            repo.createNote(note)

            useCase("n3", "new summary").getOrThrow()

            val updated = repo.getNoteById("n3", testUserId)

            assertEquals("new summary", updated?.summary)
        }

    @Test
    fun applySummary_throwsIfNoteNotFound() =
        runBlocking {
            val repo = FakeNotesRepo()
            val useCase = ApplySummaryUseCase(repo, getUserIdUsecase)

            val result = useCase("missing_id", "new summary")
            assertEquals(true, result.isFailure)
            assertEquals("Note not found", result.exceptionOrNull()?.message)
        }

    @Test
    fun applyTags_updatesNoteTags() =
        runBlocking {
            val repo = FakeNotesRepo()
            val useCase = ApplyTagsUseCase(repo, getUserIdUsecase)

            val note =
                Note(
                    id = "n4",
                    title = "Tags",
                    tags = setOf("old"),
                    userId = testUserId,
                )

            repo.createNote(note)

            val newTags = setOf("kotlin", "android", "openvino")

            useCase("n4", newTags).getOrThrow()

            val updated = repo.getNoteById("n4", testUserId)

            assertEquals(newTags, updated?.tags)
        }

    @Test
    fun applyTags_throwsIfNoteNotFound() =
        runBlocking {
            val repo = FakeNotesRepo()
            val useCase = ApplyTagsUseCase(repo, getUserIdUsecase)

            val result = useCase("missing_id", setOf("tag"))
            assertEquals(true, result.isFailure)
            assertEquals("Note not found", result.exceptionOrNull()?.message)
        }
}

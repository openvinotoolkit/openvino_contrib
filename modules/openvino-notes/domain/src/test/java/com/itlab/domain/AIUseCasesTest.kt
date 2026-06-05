package com.itlab.domain

import com.itlab.domain.ai.NoteAiService
import com.itlab.domain.ai.RewriteStyle
import com.itlab.domain.model.ContentItem
import com.itlab.domain.model.DataSource
import com.itlab.domain.model.Note
import com.itlab.domain.repository.NotesRepository
import com.itlab.domain.usecase.aiusecase.ReleaseNoteAiUseCase
import com.itlab.domain.usecase.aiusecase.RewriteNoteUseCase
import com.itlab.domain.usecase.aiusecase.SuggestImageTagsUseCase
import com.itlab.domain.usecase.aiusecase.SuggestSummaryUseCase
import com.itlab.domain.usecase.aiusecase.SuggestTagsUseCase
import com.itlab.domain.usecase.aiusecase.WarmUpNoteAiUseCase
import com.itlab.domain.usecase.noteusecase.ApplySummaryUseCase
import com.itlab.domain.usecase.noteusecase.ApplyTagsUseCase
import com.itlab.domain.usecase.noteusecase.GetUserIdUseCase
import io.mockk.MockKAnnotations
import io.mockk.every
import io.mockk.impl.annotations.MockK
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
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
        var rewriteInput: String? = null
        var rewriteMaxNewTokens: Int? = null
        var imageTagsInput: List<String> = emptyList()

        var summaryResult: String = "AI summary"
        var rewriteResult: String = "AI rewrite"
        var textTagsResult: Set<String> = setOf("text-tag-1", "text-tag-2")
        var imageTagsResult: Set<String> = setOf("image-tag-1", "image-tag-2")
        var releaseCalled: Boolean = false
        var warmUpError: Throwable? = null

        override suspend fun warmUp() {
            warmUpError?.let { throw it }
        }

        override suspend fun summarize(
            text: String,
            maxInputTokens: Int,
            maxNewTokens: Int,
        ): String {
            summaryInput = text
            return summaryResult
        }

        override suspend fun tagIMGs(img: List<String>): Set<String> {
            imageTagsInput = img
            return imageTagsResult
        }

        override suspend fun suggestTags(
            text: String,
            maxInputTokens: Int,
            maxTags: Int,
        ): Set<String> {
            textTagsInput = text
            return textTagsResult
        }

        override suspend fun rewrite(
            text: String,
            style: RewriteStyle,
            maxInputTokens: Int,
            maxNewTokens: Int,
        ): String {
            rewriteInput = text
            rewriteMaxNewTokens = maxNewTokens
            return rewriteResult
        }

        override fun release() {
            releaseCalled = true
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
    fun suggestTags_returnsTextTags_andDoesNotSendImagesToAi() =
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
            assertEquals(emptyList<String>(), ai.imageTagsInput)
            assertEquals(setOf("text-tag-1", "text-tag-2"), result.getOrThrow())
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
    fun suggestImageTags_returnsImageTagsAndSendsOnlyImagesToAi() =
        runBlocking {
            val repo = FakeNotesRepo()
            val ai =
                FakeNoteAiService().apply {
                    imageTagsResult = setOf("image-1", "image-2")
                }
            val useCase = SuggestImageTagsUseCase(ai, repo, getUserIdUsecase)

            repo.createNote(
                Note(
                    id = "n-img-tags",
                    title = "Image tags",
                    userId = testUserId,
                    contentItems =
                        listOf(
                            ContentItem.Text(text = "Text is ignored for image tags"),
                            ContentItem.Image(
                                source = DataSource(localPath = "/local/image.png"),
                                mimeType = "image/png",
                            ),
                        ),
                ),
            )

            val result = useCase("n-img-tags", maxTags = 4)

            assertEquals(null, ai.textTagsInput)
            assertEquals(listOf("/local/image.png"), ai.imageTagsInput)
            assertEquals(setOf("image-1", "image-2"), result.getOrThrow())
        }

    @Test
    fun suggestImageTags_capsGeneratedTags() =
        runBlocking {
            val repo = FakeNotesRepo()
            val ai =
                FakeNoteAiService().apply {
                    imageTagsResult = setOf("image-1", "image-2", "image-3")
                }
            val useCase = SuggestImageTagsUseCase(ai, repo, getUserIdUsecase)

            repo.createNote(
                Note(
                    id = "n-img-tags-limit",
                    title = "Image tags",
                    userId = testUserId,
                    contentItems =
                        listOf(
                            ContentItem.Image(
                                source = DataSource(remoteUrl = "https://example.com/image.jpg"),
                                mimeType = "image/jpg",
                            ),
                        ),
                ),
            )

            val result = useCase("n-img-tags-limit", maxTags = 2)

            assertEquals(setOf("image-1", "image-2"), result.getOrThrow())
        }

    @Test
    fun suggestImageTags_throwsIfNoteNotFound() =
        runBlocking {
            val repo = FakeNotesRepo()
            val ai = FakeNoteAiService()
            val useCase = SuggestImageTagsUseCase(ai, repo, getUserIdUsecase)

            val result = useCase("missing_id")
            assertEquals(true, result.isFailure)
            assertEquals("Note not found: missing_id", result.exceptionOrNull()?.message)
        }

    @Test
    fun releaseNoteAi_releasesAiService() {
        val ai = FakeNoteAiService()
        val useCase = ReleaseNoteAiUseCase(ai)

        useCase().getOrThrow()

        assertEquals(true, ai.releaseCalled)
    }

    @Test
    fun warmUpNoteAi_propagatesCancellation() =
        runBlocking {
            val ai =
                FakeNoteAiService().apply {
                    warmUpError = CancellationException("warm-up was cancelled")
                }
            val useCase = WarmUpNoteAiUseCase(ai)

            val error = runCatching<Result<Unit>> { useCase() }.exceptionOrNull()

            assertTrue(error is CancellationException)
        }

    @Test
    fun rewriteNote_usesRewriteSizedGenerationBudget() =
        runBlocking {
            val repo = FakeNotesRepo()
            val ai = FakeNoteAiService()
            val useCase = RewriteNoteUseCase(ai, repo, getUserIdUsecase)

            repo.createNote(
                Note(
                    id = "n-rewrite",
                    title = "Rewrite",
                    userId = testUserId,
                    contentItems = listOf(ContentItem.Text(text = "Long editor note")),
                ),
            )

            val result = useCase("n-rewrite")

            assertEquals("AI rewrite", result.getOrThrow())
            assertEquals("Long editor note", ai.rewriteInput)
            assertEquals(192, ai.rewriteMaxNewTokens)
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

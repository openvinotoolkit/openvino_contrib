package com.itlab.notes.ui

import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.itlab.domain.cloud.SyncCheckpointStore
import com.itlab.domain.cloud.SyncManager
import com.itlab.domain.cloud.SyncScheduler
import com.itlab.domain.model.ContentItem
import com.itlab.domain.model.Note
import com.itlab.domain.model.NoteFolder
import com.itlab.domain.model.SyncState
import com.itlab.notes.auth.NotesUserIds
import com.itlab.notes.media.withoutTextItems
import com.itlab.notes.ui.notes.ALL_DIRECTORY_ID
import com.itlab.notes.ui.notes.DirectoryItemUi
import com.itlab.notes.ui.notes.FAVORITES_DIRECTORY_ID
import com.itlab.notes.ui.notes.NoteItemUi
import com.itlab.notes.ui.notes.RECENT_DIRECTORY_ID
import com.itlab.notes.ui.notes.canCreateNotesInDirectory
import com.itlab.notes.ui.notes.coerceDirectoryNameLength
import com.itlab.notes.ui.notes.isVirtualDirectory
import com.itlab.notes.ui.toSingleLineText
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import java.util.concurrent.ConcurrentHashMap

private const val EDITOR_CLOUD_SYNC_DEBOUNCE_MS = 500L
private const val MAX_CLOUD_PUSH_ROUNDS = 4

class NotesViewModel(
    private val useCases: NotesUseCases,
    private val syncScheduler: SyncScheduler,
    private val syncManager: SyncManager,
    private val syncCheckpointStore: SyncCheckpointStore,
) : ViewModel(),
    NotesViewModelContract {
    override var uiState: NotesUiState by mutableStateOf(
        NotesUiState(screen = NotesUiScreen.Directories),
    )
        private set
    private var notesJob: Job? = null
    private var latestFolders: List<NoteFolder> = emptyList()
    private var latestNotes: List<Note> = emptyList()
    private val persistMutexByNoteId = ConcurrentHashMap<String, Mutex>()
    private val editorCloudMutex = Mutex()
    private var editorPersistJob: Job? = null
    private val cloudUploadJobs = ConcurrentHashMap<String, Job>()
    private val initialFullSyncMutex = Mutex()
    private var initialFullSyncJob: Job? = null
    private val lastCloudHashByNoteId = mutableMapOf<String, Int>()
    private var editorHasLocalChanges = false
    private val aiController =
        NotesAiController(
            scope = viewModelScope,
            useCases = useCases,
            getUiState = { uiState },
            setUiState = { uiState = it },
            upsertEditorNote = ::upsertEditorNote,
            updateEditorNote = ::updateEditorNote,
        )

    init {
        viewModelScope.launch {
            useCases.observeFoldersUseCase().collect { folders ->
                latestFolders = folders
                recomputeDirectories()
            }
        }

        viewModelScope.launch {
            useCases.observeNotesByFolderUseCase(null).collect { notes ->
                latestNotes = notes
                recomputeDirectories()
            }
        }

        viewModelScope.launch {
            useCases.getUserIdUseCase()?.let { ensureInitialFullSync(it) }
        }
        aiController.warmUp()
    }

    fun ensureInitialFullSyncForCurrentUser() {
        val userId = useCases.getUserIdUseCase() ?: return
        ensureInitialFullSync(userId)
    }

    fun ensureInitialFullSync(userId: String) {
        if (initialFullSyncJob?.isActive == true) return
        initialFullSyncJob =
            viewModelScope.launch {
                initialFullSyncMutex.withLock {
                    if (syncCheckpointStore.hasCompletedInitialFullSync(userId)) return@launch
                    syncManager.syncFull(userId)
                    syncCheckpointStore.markInitialFullSyncCompleted(userId)
                }
            }
    }

    override fun onEvent(event: NotesUiEvent) {
        when (event) {
            is NotesUiEvent.OpenDirectory -> {
                aiController.cancelGeneration()
                openDirectory(event.directory)
            }
            NotesUiEvent.BackToDirectories -> {
                aiController.cancelGeneration()
                backToDirectories()
            }
            is NotesUiEvent.OpenNote -> {
                aiController.cancelGeneration()
                openNote(event.note)
            }
            NotesUiEvent.CreateNote -> {
                aiController.cancelGeneration()
                createNote()
            }
            is NotesUiEvent.CreateDirectory -> {
                val normalized =
                    event.name
                        .toSingleLineText()
                        .trim()
                        .coerceDirectoryNameLength()
                if (normalized.isNotBlank()) {
                    viewModelScope.launch {
                        useCases.createFolderUseCase(
                            NoteFolder(
                                useCases.getUserIdUseCase() ?: NotesUserIds.LOCAL_OFFLINE,
                                name = normalized,
                            ),
                        )
                        scheduleCloudSync()
                    }
                }
            }
            is NotesUiEvent.RenameDirectory -> renameDirectory(event)
            is NotesUiEvent.DeleteDirectory -> deleteDirectory(event.directoryId)
            is NotesUiEvent.MoveNoteToDirectory -> {
                if (isVirtualDirectory(event.targetDirectoryId)) return
                viewModelScope.launch {
                    useCases.moveNoteToFolderUseCase(
                        folderId = event.targetDirectoryId,
                        noteId = event.noteId,
                    )
                }
            }
            is NotesUiEvent.ToggleNoteFavorite -> toggleNoteFavorite(event.noteId)
            NotesUiEvent.BackToDirectoryNotes -> {
                aiController.cancelGeneration()
                backToDirectoryNotes()
            }
            is NotesUiEvent.LeaveEditor -> {
                aiController.cancelGeneration()
                leaveEditor(event.note)
            }
            is NotesUiEvent.PersistNote -> persistNote(event.note)
            is NotesUiEvent.SuggestSummary -> aiController.suggestAi(event.note, AiSuggestion.Summary)
            is NotesUiEvent.SuggestTags -> aiController.suggestAi(event.note, AiSuggestion.Tags)
            is NotesUiEvent.SuggestImageTags -> aiController.suggestImageTags(event.note)
            is NotesUiEvent.RewriteNote -> aiController.suggestAi(event.note, AiSuggestion.Rewrite)
            NotesUiEvent.CancelAiGeneration -> aiController.cancelGeneration()
            is NotesUiEvent.DeleteNote -> {
                viewModelScope.launch {
                    useCases.deleteNoteUseCase(event.noteId)
                }
            }
            is NotesUiEvent.NotesSearchQueryChanged -> onNotesSearchQueryChanged(event.query)
            is NotesUiEvent.DirectorySearchQueryChanged -> {
                uiState = uiState.copy(directorySearchQuery = event.query)
            }
            NotesUiEvent.SyncCloud -> syncFromPullToRefresh()
        }
    }

    fun scheduleCloudSync() {
        val userId = useCases.getUserIdUseCase() ?: return
        syncScheduler.scheduleSync(userId)
    }

    private fun syncFromPullToRefresh() {
        val userId = useCases.getUserIdUseCase() ?: return
        viewModelScope.launch {
            uiState = uiState.copy(isCloudDownloadActive = true)
            try {
                if (!syncCheckpointStore.hasCompletedInitialFullSync(userId)) {
                    initialFullSyncMutex.withLock {
                        if (!syncCheckpointStore.hasCompletedInitialFullSync(userId)) {
                            syncManager.syncFull(userId)
                            syncCheckpointStore.markInitialFullSyncCompleted(userId)
                        }
                    }
                    return@launch
                }
                if (syncManager.hasPendingLocalChanges(userId)) {
                    pushAllLocalChangesUntilIdle(userId)
                }
                syncManager.pullUpdates(userId)
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                scheduleCloudSync()
            } finally {
                uiState = uiState.copy(isCloudDownloadActive = false)
            }
        }
    }

    private suspend fun pushAllLocalChangesUntilIdle(userId: String) {
        var rounds = 0
        while (syncManager.hasPendingLocalChanges(userId) && rounds < MAX_CLOUD_PUSH_ROUNDS) {
            syncManager.pushLocalChanges(userId)
            rounds++
        }
    }

    private fun onNotesSearchQueryChanged(query: String) {
        val directory = (uiState.screen as? NotesUiScreen.DirectoryNotes)?.directory ?: return
        uiState = uiState.copy(notesSearchQuery = query)
        startNotesCollection(directory, query)
    }

    private fun renameDirectory(event: NotesUiEvent.RenameDirectory) {
        val normalized =
            event.newName
                .toSingleLineText()
                .trim()
                .coerceDirectoryNameLength()
        if (normalized.isBlank() || isVirtualDirectory(event.directoryId)) return
        viewModelScope.launch {
            val existingFolder = useCases.getFolderUseCase(event.directoryId) ?: return@launch
            useCases.updateFolderUseCase(existingFolder.copy(name = normalized))
        }
    }

    private fun deleteDirectory(directoryId: String) {
        if (isVirtualDirectory(directoryId)) return
        viewModelScope.launch {
            useCases.deleteFolderUseCase(directoryId)
            if ((uiState.screen as? NotesUiScreen.DirectoryNotes)?.directory?.id == directoryId) {
                backToDirectories()
            }
        }
    }

    private fun openDirectory(directory: DirectoryItemUi) {
        uiState =
            uiState.copy(
                screen = NotesUiScreen.DirectoryNotes(directory = directory),
                notes = emptyList(),
                notesSearchQuery = "",
                aiState = aiController.freshAiState(),
                imageTaggingState = ImageTaggingUiState(),
            )
        startNotesCollection(directory, searchQuery = "")
    }

    private fun startNotesCollection(
        directory: DirectoryItemUi,
        searchQuery: String,
    ) {
        notesJob?.cancel()
        notesJob =
            viewModelScope.launch {
                notesFlowForDirectory(useCases, latestFolders, directory, searchQuery).collect { notes ->
                    val opened = uiState.screen as? NotesUiScreen.DirectoryNotes ?: return@collect
                    uiState =
                        uiState.copy(
                            notes = notes.map { it.toUi() },
                            notesSearchQuery = searchQuery,
                            screen =
                                NotesUiScreen.DirectoryNotes(
                                    directory = opened.directory.copy(noteCount = notes.size),
                                ),
                        )
                }
            }
    }

    private val backToDirectories: () -> Unit = {
        uiState =
            uiState.copy(
                screen = NotesUiScreen.Directories,
                notes = emptyList(),
                notesSearchQuery = "",
                aiState = aiController.freshAiState(),
                imageTaggingState = ImageTaggingUiState(),
            )
    }

    private fun openNote(note: NoteItemUi) {
        val dir = (uiState.screen as? NotesUiScreen.DirectoryNotes)?.directory ?: return
        notesJob?.cancel()
        viewModelScope.launch {
            val enriched =
                useCases.getNoteUseCase(note.id)?.toUi()
                    ?: note
            resetEditorCloudBaseline(enriched)
            uiState =
                uiState.copy(
                    screen = NotesUiScreen.NoteEditor(directory = dir, note = enriched),
                    aiState = aiController.freshAiState(),
                    imageTaggingState = ImageTaggingUiState(),
                )
            aiController.warmUp()
        }
    }

    private fun createNote() {
        val dir = (uiState.screen as? NotesUiScreen.DirectoryNotes)?.directory ?: return
        if (!canCreateNotesInDirectory(dir.id)) return
        notesJob?.cancel()
        val userId = useCases.getUserIdUseCase() ?: NotesUserIds.LOCAL_OFFLINE
        val newNote = Note(userId = userId, folderId = dir.id.asDomainFolderId()).toUi()
        resetEditorCloudBaseline(newNote)
        uiState =
            uiState.copy(
                screen = NotesUiScreen.NoteEditor(directory = dir, note = newNote),
                aiState = aiController.freshAiState(),
                imageTaggingState = ImageTaggingUiState(),
            )
        aiController.warmUp()
    }

    private fun resetEditorCloudBaseline(note: NoteItemUi) {
        lastCloudHashByNoteId[note.id] = note.contentCloudHash()
        editorHasLocalChanges = false
    }

    private fun backToDirectoryNotes() {
        val editor = uiState.screen as? NotesUiScreen.NoteEditor ?: return
        val directory = editor.directory
        uiState =
            uiState.copy(
                screen = NotesUiScreen.DirectoryNotes(directory = directory),
                aiState = aiController.freshAiState(),
                imageTaggingState = ImageTaggingUiState(),
            )
        startNotesCollection(directory, uiState.notesSearchQuery)
    }

    private fun toggleNoteFavorite(noteId: String) {
        viewModelScope.launch {
            useCases.switchFavoriteUseCase(noteId)
            val editor = uiState.screen as? NotesUiScreen.NoteEditor
            if (editor?.note?.id == noteId) {
                uiState =
                    uiState.copy(
                        screen =
                            editor.copy(
                                note = editor.note.copy(isFavorite = !editor.note.isFavorite),
                            ),
                    )
            }
        }
    }

    private fun persistNote(note: NoteItemUi) {
        val editor = uiState.screen as? NotesUiScreen.NoteEditor ?: return
        val directory = editor.directory
        val previousPersistJob = editorPersistJob
        editorPersistJob =
            viewModelScope.launch {
                previousPersistJob?.join()
                if (!persistNoteToRepository(note, directory)) return@launch
                markEditorChangedIfNeeded(note)
                val userId = useCases.getUserIdUseCase() ?: return@launch
                if (
                    editorHasLocalChanges ||
                    syncManager.hasPendingLocalChangesForNote(userId, note.id)
                ) {
                    scheduleCloudUploadForNote(note, directory)
                }
            }
    }

    private fun leaveEditor(note: NoteItemUi) {
        val editor = uiState.screen as? NotesUiScreen.NoteEditor ?: return
        val directory = editor.directory
        viewModelScope.launch {
            editorPersistJob?.join()
            if (note.title.trim().isNotEmpty()) {
                val saved = persistNoteToRepository(note, directory)
                if (saved) {
                    markEditorChangedIfNeeded(note)
                }
            }
            val userId = useCases.getUserIdUseCase()
            if (
                editorHasLocalChanges ||
                (userId != null && syncManager.hasPendingLocalChangesForNote(userId, note.id))
            ) {
                scheduleCloudUploadForNote(note, directory)
            }
            navigateBackToDirectoryNotes(directory)
        }
    }

    private fun scheduleCloudUploadForNote(
        note: NoteItemUi,
        directory: DirectoryItemUi,
    ) {
        val userId = useCases.getUserIdUseCase() ?: return

        cloudUploadJobs[note.id]?.cancel()
        cloudUploadJobs[note.id] =
            viewModelScope.launch {
                delay(EDITOR_CLOUD_SYNC_DEBOUNCE_MS)
                if (!editorHasLocalChanges && !syncManager.hasPendingLocalChangesForNote(userId, note.id)) {
                    return@launch
                }
                setNoteUploading(note.id, uploading = true)
                try {
                    pushNoteToCloudUntilComplete(userId, note, directory)
                    onEditorCloudPushSucceeded(note)
                    setEditorCloudSyncStatus(EditorCloudSyncStatus.Idle)
                } catch (e: CancellationException) {
                    throw e
                } catch (e: Exception) {
                    setEditorCloudSyncStatus(EditorCloudSyncStatus.Error)
                    scheduleCloudSync()
                } finally {
                    setNoteUploading(note.id, uploading = false)
                    cloudUploadJobs.remove(note.id)
                }
            }
    }

    private suspend fun pushNoteToCloudUntilComplete(
        userId: String,
        note: NoteItemUi,
        directory: DirectoryItemUi,
    ) {
        editorCloudMutex.withLock {
            if (!syncCheckpointStore.hasCompletedInitialFullSync(userId)) {
                initialFullSyncMutex.withLock {
                    if (!syncCheckpointStore.hasCompletedInitialFullSync(userId)) {
                        syncManager.syncFull(userId)
                        syncCheckpointStore.markInitialFullSyncCompleted(userId)
                    }
                }
            }
            if (note.title.trim().isNotEmpty()) {
                persistNoteToRepositoryLocked(note, directory)
            }
            var rounds = 0
            while (
                syncManager.hasPendingLocalChangesForNote(userId, note.id) &&
                rounds < MAX_CLOUD_PUSH_ROUNDS
            ) {
                syncManager.pushLocalChanges(userId)
                rounds++
            }
        }
    }

    private fun setNoteUploading(
        noteId: String,
        uploading: Boolean,
    ) {
        val nextIds =
            if (uploading) {
                uiState.noteIdsUploading + noteId
            } else {
                uiState.noteIdsUploading - noteId
            }
        uiState = uiState.copy(noteIdsUploading = nextIds)
        val editor = uiState.screen as? NotesUiScreen.NoteEditor
        if (editor?.note?.id == noteId) {
            uiState =
                uiState.copy(
                    screen =
                        editor.copy(
                            cloudSyncStatus =
                                if (uploading) {
                                    EditorCloudSyncStatus.Uploading
                                } else {
                                    EditorCloudSyncStatus.Idle
                                },
                        ),
                )
        }
    }

    private fun markEditorChangedIfNeeded(note: NoteItemUi) {
        val hash = note.contentCloudHash()
        val baseline = lastCloudHashByNoteId[note.id]
        if (baseline == null) {
            lastCloudHashByNoteId[note.id] = hash
            return
        }
        if (hash != baseline) {
            editorHasLocalChanges = true
        }
    }

    private fun onEditorCloudPushSucceeded(note: NoteItemUi) {
        lastCloudHashByNoteId[note.id] = note.contentCloudHash()
        editorHasLocalChanges = false
    }

    private fun setEditorCloudSyncStatus(status: EditorCloudSyncStatus) {
        val editor = uiState.screen as? NotesUiScreen.NoteEditor ?: return
        if (editor.cloudSyncStatus == status) return
        uiState = uiState.copy(screen = editor.copy(cloudSyncStatus = status))
    }

    fun isNoteUploading(noteId: String): Boolean = noteId in uiState.noteIdsUploading

    private fun navigateBackToDirectoryNotes(directory: DirectoryItemUi) {
        uiState =
            uiState.copy(
                screen = NotesUiScreen.DirectoryNotes(directory = directory),
                aiState = aiController.freshAiState(),
                imageTaggingState = ImageTaggingUiState(),
            )
        startNotesCollection(directory, uiState.notesSearchQuery)
    }

    private suspend fun persistNoteToRepository(
        note: NoteItemUi,
        directory: DirectoryItemUi,
    ): Boolean {
        val mutex = persistMutexByNoteId.getOrPut(note.id) { Mutex() }
        return mutex.withLock {
            persistNoteToRepositoryLocked(note, directory)
        }
    }

    private suspend fun persistNoteToRepositoryLocked(
        note: NoteItemUi,
        directory: DirectoryItemUi,
    ): Boolean {
        if (note.title.trim().isEmpty()) return false
        if (!canCreateNotesInDirectory(directory.id)) {
            val existing = useCases.getNoteUseCase(note.id)
            if (existing == null) return false
        }
        val authUserId = useCases.getUserIdUseCase()
        val targetFolderId = note.folderId ?: directory.id.asDomainFolderId()
        val domainNote =
            note
                .toDomain(folderId = targetFolderId)
                .let { draft ->
                    if (authUserId != null) draft.copy(userId = authUserId) else draft
                }
        val existing = useCases.getNoteUseCase(note.id)
        val persistedId =
            if (existing != null) {
                val updateResult = useCases.updateNoteUseCase(existing.applyUiUpdate(note, targetFolderId))
                if (updateResult.isFailure) return false
                note.id
            } else {
                val createResult = useCases.createNoteUseCase(domainNote)
                createResult.getOrElse { return false }
            }
        val savedNote =
            note.copy(
                id = persistedId,
                userId = authUserId ?: note.userId,
                folderId = targetFolderId,
            )
        val refreshedNote = useCases.getNoteUseCase(persistedId)?.toUi() ?: savedNote
        val editorNote =
            refreshedNote.copy(
                content = savedNote.content,
            )
        val editor = uiState.screen as? NotesUiScreen.NoteEditor
        if (editor != null && (editor.note.id == note.id || editor.note.id == persistedId)) {
            uiState = uiState.copy(screen = editor.copy(note = editorNote))
        }
        return true
    }

    private suspend fun upsertEditorNote(
        note: NoteItemUi,
        directory: DirectoryItemUi,
    ): Result<NoteItemUi> =
        runCatching {
            require(note.title.trim().isNotEmpty()) { "Title is required" }
            val targetFolderId = note.folderId ?: directory.id.asDomainFolderId()
            val authUserId = useCases.getUserIdUseCase()
            val existing = useCases.getNoteUseCase(note.id)
            if (existing != null) {
                useCases.updateNoteUseCase(existing.applyUiUpdate(note, targetFolderId)).getOrThrow()
                note.copy(
                    userId = authUserId ?: note.userId,
                    folderId = targetFolderId,
                )
            } else {
                val domainNote =
                    note
                        .toDomain(folderId = targetFolderId)
                        .let { draft ->
                            if (authUserId != null) draft.copy(userId = authUserId) else draft
                        }
                val savedId = useCases.createNoteUseCase(domainNote).getOrThrow()
                note.copy(
                    id = savedId,
                    userId = authUserId ?: note.userId,
                    folderId = targetFolderId,
                )
            }
        }

    private fun updateEditorNote(note: NoteItemUi) {
        val editor = uiState.screen as? NotesUiScreen.NoteEditor ?: return
        uiState =
            uiState.copy(
                screen = editor.copy(note = note),
            )
    }

    private fun recomputeDirectories() {
        val directories = buildDirectoryItems(latestFolders, latestNotes)
        uiState =
            uiState.copy(
                directories = directories,
                screen = uiState.screen.withUpdatedDirectoryCount(directories),
            )
    }

    override fun onCleared() {
        notesJob?.cancel()
        aiController.cancelAll()
        super.onCleared()
    }
}

internal fun NoteFolder.toUi(noteCount: Int): DirectoryItemUi =
    DirectoryItemUi(id = id, name = name, noteCount = noteCount)

internal fun Note.toUi(): NoteItemUi =
    NoteItemUi(
        id = id,
        userId = userId,
        title = title,
        content =
            contentItems
                .filterIsInstance<ContentItem.Text>()
                .joinToString("\n") { it.text },
        folderId = folderId,
        attachments = contentItems.withoutTextItems(),
        isFavorite = isFavorite,
        tags = tags,
        summary = summary,
    )

internal fun NoteItemUi.toContentItems(): List<ContentItem> =
    buildList {
        if (content.isNotBlank()) add(ContentItem.Text(text = content))
        addAll(attachments.withoutTextItems())
    }

internal fun NoteItemUi.toDomain(folderId: String?): Note =
    Note(
        userId = userId,
        id = id,
        title = title,
        folderId = folderId,
        contentItems = toContentItems(),
        isFavorite = isFavorite,
        tags = tags,
        summary = summary,
    )

internal fun NoteItemUi.contentCloudHash(): Int {
    var result = title.hashCode()
    result = 31 * result + content.hashCode()
    for (item in attachments.withoutTextItems()) {
        result = 31 * result + item.id.hashCode()
    }
    return result
}

internal fun Note.applyUiUpdate(
    ui: NoteItemUi,
    targetFolderId: String?,
): Note =
    copy(
        title = ui.title,
        folderId = targetFolderId,
        contentItems = ui.toContentItems(),
        isFavorite = ui.isFavorite,
        tags = ui.tags,
        summary = ui.summary,
        syncStatus = SyncState.PENDING,
    )

internal fun String.asDomainFolderId(): String? =
    when (this) {
        ALL_DIRECTORY_ID,
        RECENT_DIRECTORY_ID,
        FAVORITES_DIRECTORY_ID,
        -> null
        else -> this
    }

internal fun DirectoryItemUi.folderIdForSearch(): String? =
    when (id) {
        ALL_DIRECTORY_ID,
        RECENT_DIRECTORY_ID,
        FAVORITES_DIRECTORY_ID,
        -> null
        else -> id
    }

internal fun filterDirectoriesByName(
    directories: List<DirectoryItemUi>,
    query: String,
): List<DirectoryItemUi> {
    val normalized = query.trim()
    if (normalized.isBlank()) return directories
    return directories.filter { directory ->
        directory.name.contains(normalized, ignoreCase = true)
    }
}

package com.itlab.notes.ui.notes

import androidx.activity.compose.BackHandler
import androidx.compose.foundation.ExperimentalFoundationApi
import androidx.compose.foundation.clickable
import androidx.compose.foundation.combinedClickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.rounded.ArrowBack
import androidx.compose.material.icons.automirrored.rounded.CompareArrows
import androidx.compose.material.icons.rounded.Add
import androidx.compose.material.icons.rounded.Close
import androidx.compose.material.icons.rounded.Delete
import androidx.compose.material.icons.rounded.Folder
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CenterAlignedTopAppBar
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FloatingActionButton
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import com.itlab.notes.onboarding.OnboardingTargets
import com.itlab.notes.onboarding.onboardingTarget

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun notesListScreen(
    directoryId: String,
    directoryName: String,
    notes: List<NoteItemUi>,
    searchQuery: String,
    onSearchQueryChange: (String) -> Unit,
    directories: List<DirectoryItemUi>,
    actions: NotesListActions,
    pullToRefreshEnabled: Boolean = false,
    isPullRefreshing: Boolean = false,
    noteIdsUploading: Set<String> = emptySet(),
    onPullToRefresh: () -> Unit = {},
) {
    val colors = MaterialTheme.colorScheme
    val selectedNoteIds = remember { mutableStateListOf<String>() }
    var showMoveDialog by remember { mutableStateOf(false) }
    var showDeleteDialog by remember { mutableStateOf(false) }
    val isSelectionMode = selectedNoteIds.isNotEmpty()
    val selectedCount = selectedNoteIds.size
    val clearSelection = {
        selectedNoteIds.clear()
        showMoveDialog = false
        showDeleteDialog = false
    }
    val deleteSelected = {
        notes.filter { it.id in selectedNoteIds }.forEach { note ->
            actions.onNoteDelete(note)
        }
        clearSelection()
    }
    val handleBack = {
        when {
            showDeleteDialog -> showDeleteDialog = false
            showMoveDialog -> showMoveDialog = false
            isSelectionMode -> clearSelection()
            else -> actions.onBack()
        }
    }

    BackHandler(onBack = handleBack)

    Box(Modifier.fillMaxSize()) {
        notesPullToRefreshBox(
            enabled = pullToRefreshEnabled,
            isRefreshing = isPullRefreshing,
            onRefresh = onPullToRefresh,
            modifier = Modifier.fillMaxSize(),
        ) {
            Scaffold(
                containerColor = colors.background,
                topBar = {
                    notesTopBar(
                        directoryName = directoryName,
                        selectedCount = selectedCount,
                        onBack = handleBack,
                        onMoveSelected = { showMoveDialog = true },
                        onDeleteSelected = { showDeleteDialog = true },
                    )
                },
                floatingActionButton = {
                    if (!isSelectionMode && canCreateNotesInDirectory(directoryId)) {
                        notesFab(onAddNoteClick = actions.onAddNoteClick)
                    }
                },
            ) { paddingValues ->
                notesListContent(
                    notes = notes,
                    searchQuery = searchQuery,
                    onSearchQueryChange = onSearchQueryChange,
                    selectedNoteIds = selectedNoteIds,
                    noteIdsUploading = noteIdsUploading,
                    paddingValues = paddingValues,
                    actions =
                        NotesListContentActions(
                            onNoteDelete = actions.onNoteDelete,
                            onNoteClick = actions.onNoteClick,
                        ),
                )
            }
        }
        if (showMoveDialog && selectedNoteIds.isNotEmpty()) {
            notesMoveNotesDialog(
                currentDirectoryId = directoryId,
                directories = directories,
                onDismissRequest = { showMoveDialog = false },
                onFolderChosen = { folderId ->
                    selectedNoteIds.forEach { noteId -> actions.onNoteMove(noteId, folderId) }
                    selectedNoteIds.clear()
                    showMoveDialog = false
                },
            )
        }
        if (showDeleteDialog && selectedNoteIds.isNotEmpty()) {
            notesDeleteConfirmationDialog(
                selectedCount = selectedCount,
                onDismissRequest = { showDeleteDialog = false },
                onConfirmDelete = deleteSelected,
            )
        }
    }
}

data class NotesListActions(
    val onBack: () -> Unit,
    val onAddNoteClick: () -> Unit,
    val onNoteDelete: (NoteItemUi) -> Unit,
    val onNoteMove: (noteId: String, directoryId: String) -> Unit,
    val onNoteClick: (NoteItemUi) -> Unit,
)

private data class NotesListContentActions(
    val onNoteDelete: (NoteItemUi) -> Unit,
    val onNoteClick: (NoteItemUi) -> Unit,
)

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun notesTopBar(
    directoryName: String,
    selectedCount: Int,
    onBack: () -> Unit,
    onMoveSelected: () -> Unit,
    onDeleteSelected: () -> Unit,
) {
    val colors = MaterialTheme.colorScheme
    CenterAlignedTopAppBar(
        title = {
            Text(
                text = if (selectedCount > 0) "$selectedCount selected" else directoryName,
                color = colors.onSurface,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis,
                modifier = Modifier.fillMaxWidth(),
            )
        },
        navigationIcon = {
            IconButton(onClick = onBack) {
                Icon(
                    imageVector =
                        if (selectedCount > 0) {
                            Icons.Rounded.Close
                        } else {
                            Icons.AutoMirrored.Rounded.ArrowBack
                        },
                    contentDescription = null,
                    tint = colors.onSurface,
                )
            }
        },
        actions = {
            if (selectedCount > 0) {
                IconButton(onClick = onMoveSelected) {
                    Icon(
                        imageVector = Icons.AutoMirrored.Rounded.CompareArrows,
                        contentDescription = null,
                        tint = colors.onSurface,
                    )
                }
                IconButton(onClick = onDeleteSelected) {
                    Icon(
                        imageVector = Icons.Rounded.Delete,
                        contentDescription = null,
                        tint = colors.onSurface,
                    )
                }
            }
        },
        colors =
            TopAppBarDefaults.topAppBarColors(
                containerColor = Color.Transparent,
                scrolledContainerColor = Color.Unspecified,
                navigationIconContentColor = Color.Unspecified,
                titleContentColor = Color.Unspecified,
                actionIconContentColor = Color.Unspecified,
            ),
    )
}

@Composable
private fun notesMoveNotesDialog(
    currentDirectoryId: String,
    directories: List<DirectoryItemUi>,
    onDismissRequest: () -> Unit,
    onFolderChosen: (String) -> Unit,
) {
    val moveTargets =
        remember(directories, currentDirectoryId) {
            directories.filter { !isVirtualDirectory(it.id) && it.id != currentDirectoryId }
        }
    universalBasicAlertDialog(
        onDismissRequest = onDismissRequest,
        slots =
            UniversalBasicAlertDialogSlots(
                icon = Icons.AutoMirrored.Rounded.CompareArrows,
                iconContainerColor = MaterialTheme.colorScheme.surfaceContainer,
                iconTintColor = MaterialTheme.colorScheme.onSurfaceVariant,
                title = {
                    Text(
                        text = "Move to folder",
                        fontWeight = FontWeight.W400,
                    )
                },
                input = {
                    notesMoveTargetsBlock(
                        directories = moveTargets,
                        onFolderChosen = onFolderChosen,
                    )
                },
                actions = {
                    TextButton(
                        onClick = onDismissRequest,
                        contentPadding = PaddingValues(horizontal = 12.dp),
                    ) {
                        Text("Cancel")
                    }
                },
            ),
    )
}

@Composable
private fun notesMoveTargetsBlock(
    directories: List<DirectoryItemUi>,
    onFolderChosen: (String) -> Unit,
) {
    Surface(
        color = MaterialTheme.colorScheme.surfaceContainer,
        shape = MaterialTheme.shapes.large,
        modifier = Modifier.fillMaxWidth(),
    ) {
        if (directories.isEmpty()) {
            Box(
                modifier =
                    Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 16.dp, vertical = 20.dp),
                contentAlignment = Alignment.Center,
            ) {
                Text(
                    text = "No other folders to move to",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        } else {
            Column(
                modifier =
                    Modifier
                        .fillMaxWidth()
                        .heightIn(max = 180.dp)
                        .verticalScroll(rememberScrollState()),
            ) {
                directories.forEachIndexed { index, dir ->
                    notesMoveTargetRow(
                        directory = dir,
                        onClick = { onFolderChosen(dir.id) },
                        modifier = Modifier.padding(horizontal = 12.dp, vertical = 0.dp),
                    )
                    if (index < directories.lastIndex) {
                        notesMoveTargetsDivider()
                    }
                }
            }
        }
    }
}

@Composable
private fun notesMoveTargetRow(
    directory: DirectoryItemUi,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
) {
    val colors = MaterialTheme.colorScheme
    Row(
        modifier =
            modifier
                .fillMaxWidth()
                .clip(MaterialTheme.shapes.medium)
                .clickable(onClick = onClick)
                .padding(horizontal = 12.dp, vertical = 11.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Icon(
            imageVector = Icons.Rounded.Folder,
            contentDescription = null,
            tint = colors.onSurfaceVariant,
            modifier = Modifier.size(25.dp),
        )
        Spacer(Modifier.width(12.dp))
        Text(
            text = directory.name,
            color = colors.onSurface,
            style = MaterialTheme.typography.bodyLarge,
            modifier = Modifier.weight(1f),
            maxLines = 1,
            overflow = TextOverflow.Ellipsis,
        )
    }
}

@Composable
private fun notesMoveTargetsDivider() {
    Box(
        modifier = Modifier.fillMaxWidth(),
        contentAlignment = Alignment.Center,
    ) {
        Box(
            modifier =
                Modifier
                    .padding(horizontal = 10.dp)
                    .fillMaxWidth(0.9f),
            contentAlignment = Alignment.Center,
        ) {
            HorizontalDivider(
                color = MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.55f),
                thickness = 1.dp,
            )
        }
    }
}

@Composable
private fun notesDeleteConfirmationDialog(
    selectedCount: Int,
    onDismissRequest: () -> Unit,
    onConfirmDelete: () -> Unit,
) {
    universalBasicAlertDialog(
        onDismissRequest = onDismissRequest,
        slots =
            UniversalBasicAlertDialogSlots(
                icon = Icons.Rounded.Delete,
                iconContainerColor = MaterialTheme.colorScheme.errorContainer,
                iconTintColor = MaterialTheme.colorScheme.onErrorContainer,
                title = {
                    Text(
                        text = "Delete selected notes?",
                    )
                },
                input = {
                    Text(
                        text = "This will permanently delete $selectedCount note(s).",
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        style = MaterialTheme.typography.bodyMedium,
                    )
                },
                actions = {
                    TextButton(
                        onClick = onDismissRequest,
                        contentPadding = PaddingValues(horizontal = 12.dp),
                    ) {
                        Text("Cancel")
                    }
                    Button(
                        onClick = onConfirmDelete,
                        contentPadding = PaddingValues(horizontal = 16.dp),
                        shape = MaterialTheme.shapes.medium,
                        colors =
                            ButtonDefaults.buttonColors(
                                containerColor = MaterialTheme.colorScheme.error,
                                contentColor = MaterialTheme.colorScheme.onError,
                            ),
                    ) {
                        Text("Delete")
                    }
                },
            ),
    )
}

@Composable
private fun notesFab(onAddNoteClick: () -> Unit) {
    val colors = MaterialTheme.colorScheme
    FloatingActionButton(
        onClick = onAddNoteClick,
        modifier = Modifier.onboardingTarget(OnboardingTargets.NOTES_FAB),
        containerColor = colors.primary,
    ) {
        Icon(
            Icons.Rounded.Add,
            contentDescription = null,
            tint = colors.onPrimary,
        )
    }
}

@Composable
private fun notesListContent(
    notes: List<NoteItemUi>,
    searchQuery: String,
    onSearchQueryChange: (String) -> Unit,
    selectedNoteIds: MutableList<String>,
    noteIdsUploading: Set<String>,
    paddingValues: PaddingValues,
    actions: NotesListContentActions,
) {
    val isSelectionMode = selectedNoteIds.isNotEmpty()
    Column(
        modifier =
            Modifier
                .fillMaxSize()
                .padding(paddingValues)
                .padding(horizontal = 16.dp),
    ) {
        searchField(
            query = searchQuery,
            onQueryChange = onSearchQueryChange,
        )

        val isSearchActive = searchQuery.isNotBlank()

        BoxWithConstraints(
            modifier =
                Modifier
                    .weight(1f)
                    .fillMaxWidth(),
        ) {
            LazyColumn(
                verticalArrangement = Arrangement.spacedBy(12.dp),
                modifier =
                    Modifier
                        .fillMaxSize()
                        .heightIn(min = maxHeight + 1.dp)
                        .padding(top = 4.dp),
            ) {
                if (notes.isEmpty() && isSearchActive) {
                    item {
                        Box(
                            modifier =
                                Modifier
                                    .fillMaxWidth()
                                    .padding(top = 80.dp)
                                    .heightIn(min = 220.dp),
                            contentAlignment = Alignment.Center,
                        ) {
                            notesSearchEmptyState()
                        }
                    }
                }
                val tourNoteId = notes.firstOrNull()?.id
                items(
                    items = notes,
                    key = { note -> note.id },
                ) { note ->
                    notesListItem(
                        note = note,
                        isSelected = note.id in selectedNoteIds,
                        isUploadingToCloud = note.id in noteIdsUploading,
                        modifier =
                            if (note.id == tourNoteId) {
                                Modifier.onboardingTarget(OnboardingTargets.NOTES_NOTE_ROW)
                            } else {
                                Modifier
                            },
                        onClick = {
                            if (isSelectionMode) {
                                if (note.id in selectedNoteIds) {
                                    selectedNoteIds.remove(note.id)
                                } else {
                                    selectedNoteIds.add(note.id)
                                }
                            } else {
                                actions.onNoteClick(note)
                            }
                        },
                        onLongClick = {
                            if (note.id !in selectedNoteIds) {
                                selectedNoteIds.add(note.id)
                            }
                        },
                    )
                }
            }
        }
    }
}

@Composable
private fun notesListItem(
    note: NoteItemUi,
    isSelected: Boolean,
    isUploadingToCloud: Boolean,
    modifier: Modifier = Modifier,
    onClick: () -> Unit,
    onLongClick: () -> Unit,
) {
    noteCard(
        note = note,
        isSelected = isSelected,
        isUploadingToCloud = isUploadingToCloud,
        modifier = modifier,
        onClick = onClick,
        onLongClick = onLongClick,
    )
}

@Composable
@OptIn(ExperimentalFoundationApi::class)
private fun noteCard(
    note: NoteItemUi,
    isSelected: Boolean,
    isUploadingToCloud: Boolean,
    modifier: Modifier = Modifier,
    onClick: () -> Unit,
    onLongClick: () -> Unit,
) {
    val colors = MaterialTheme.colorScheme

    Card(
        colors =
            CardDefaults.cardColors(
                containerColor =
                    if (isSelected) {
                        colors.surfaceContainerHighest
                    } else {
                        colors.surfaceContainer
                    },
            ),
        shape = MaterialTheme.shapes.large,
        modifier =
            modifier
                .fillMaxWidth()
                .clip(MaterialTheme.shapes.large)
                .combinedClickable(
                    onClick = onClick,
                    onLongClick = onLongClick,
                ),
    ) {
        Column(modifier = Modifier.padding(16.dp).fillMaxWidth()) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Text(
                    text = note.title,
                    modifier = Modifier.weight(1f),
                    color =
                        if (isSelected) {
                            colors.onPrimaryContainer
                        } else {
                            colors.onSurface
                        },
                    style = MaterialTheme.typography.titleMedium,
                    maxLines = 2,
                    overflow = TextOverflow.Ellipsis,
                )
                if (isUploadingToCloud) {
                    Spacer(Modifier.width(8.dp))
                    CircularProgressIndicator(
                        modifier = Modifier.size(18.dp),
                        strokeWidth = 2.dp,
                        color = colors.primary,
                    )
                }
            }
            Spacer(Modifier.height(8.dp))
            Text(
                text = noteCardDescriptionText(note),
                color =
                    if (isSelected) {
                        colors.onPrimaryContainer
                    } else if (note.content.isBlank()) {
                        colors.onSurfaceVariant.copy(alpha = 0.7f)
                    } else {
                        colors.onSurfaceVariant
                    },
                style = MaterialTheme.typography.bodySmall,
                maxLines = 2,
                overflow = TextOverflow.Ellipsis,
            )
        }
    }
}

private fun noteCardDescriptionText(note: NoteItemUi): String =
    buildString {
        append(if (note.content.isNotBlank()) note.content else "No description")
        if (note.attachments.isNotEmpty()) {
            append(" · ")
            append(note.attachments.size)
            append(" attachment")
            if (note.attachments.size != 1) append("s")
        }
    }

@Composable
private fun searchField(
    query: String,
    onQueryChange: (String) -> Unit,
) {
    appSearchField(
        value = query,
        onValueChange = onQueryChange,
        modifier =
            Modifier
                .padding(vertical = 16.dp)
                .onboardingTarget(OnboardingTargets.NOTES_SEARCH),
        placeholderText = "Search notes",
    )
}

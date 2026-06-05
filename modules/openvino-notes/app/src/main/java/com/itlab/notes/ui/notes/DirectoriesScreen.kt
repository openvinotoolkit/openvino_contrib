@file:Suppress("TooManyFunctions")

package com.itlab.notes.ui.notes

import androidx.activity.compose.BackHandler
import androidx.compose.foundation.ExperimentalFoundationApi
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.combinedClickable
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.RowScope
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.sizeIn
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.LazyListScope
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.rounded.Login
import androidx.compose.material.icons.automirrored.rounded.Logout
import androidx.compose.material.icons.rounded.Add
import androidx.compose.material.icons.rounded.AllInbox
import androidx.compose.material.icons.rounded.ChevronRight
import androidx.compose.material.icons.rounded.Edit
import androidx.compose.material.icons.rounded.Folder
import androidx.compose.material.icons.rounded.FolderCopy
import androidx.compose.material.icons.rounded.Schedule
import androidx.compose.material.icons.rounded.Star
import androidx.compose.material.icons.rounded.TextFields
import androidx.compose.material3.BasicAlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.CenterAlignedTopAppBar
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LocalContentColor
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextFieldDefaults
import androidx.compose.material3.ProvideTextStyle
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.focus.FocusManager
import androidx.compose.ui.focus.FocusRequester
import androidx.compose.ui.focus.focusRequester
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalFocusManager
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.input.KeyboardCapitalization
import androidx.compose.ui.text.input.VisualTransformation
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.DialogProperties
import com.itlab.notes.onboarding.OnboardingTargets
import com.itlab.notes.onboarding.onboardingTarget
import com.itlab.notes.ui.toSingleLineText

private const val DIRECTORY_NAME_TAKEN_ERROR = "A directory with this name already exists"

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun directoriesScreen(
    directories: List<DirectoryItemUi>,
    searchQuery: String,
    onSearchQueryChange: (String) -> Unit,
    onCreateDirectory: (String) -> Unit,
    onDeleteDirectory: (DirectoryItemUi) -> Unit,
    onRenameDirectory: (DirectoryItemUi, String) -> Unit,
    onDirectoryClick: (DirectoryItemUi) -> Unit,
    showSignOut: Boolean = false,
    onSignOut: () -> Unit = {},
    pullToRefreshEnabled: Boolean = false,
    isPullRefreshing: Boolean = false,
    onPullToRefresh: () -> Unit = {},
    showReturnToSignIn: Boolean = false,
    onReturnToSignIn: () -> Unit = {},
) {
    val colors = MaterialTheme.colorScheme
    val focusManager = LocalFocusManager.current
    var showCreateDialog by remember { mutableStateOf(false) }

    BackHandler(enabled = showCreateDialog) {
        showCreateDialog = false
    }

    notesPullToRefreshBox(
        enabled = pullToRefreshEnabled,
        isRefreshing = isPullRefreshing,
        onRefresh = onPullToRefresh,
        modifier = Modifier.fillMaxSize(),
    ) {
        Scaffold(
            modifier =
                Modifier
                    .fillMaxSize()
                    .clickable(
                        interactionSource = remember { MutableInteractionSource() },
                        indication = null,
                    ) {
                        focusManager.clearFocus(force = true)
                    },
            containerColor = colors.background,
            topBar = {
                directoriesTopBar(
                    showSignOut = showSignOut,
                    onSignOut = onSignOut,
                    showReturnToSignIn = showReturnToSignIn,
                    onReturnToSignIn = onReturnToSignIn,
                    onAddDirectoryClick = { showCreateDialog = true },
                )
            },
        ) { paddingValues ->
            directoriesList(
                directories = directories,
                searchQuery = searchQuery,
                onSearchQueryChange = onSearchQueryChange,
                onDirectoryLongClick = onDeleteDirectory,
                onDirectoryRename = onRenameDirectory,
                onDirectoryClick = onDirectoryClick,
                modifier =
                    Modifier
                        .fillMaxSize()
                        .padding(paddingValues),
            )
        }
    }
    if (showCreateDialog) {
        directoriesCreateDirectoryDialog(
            directories = directories,
            onDismissRequest = { showCreateDialog = false },
            onCreateDirectory = onCreateDirectory,
        )
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun directoriesCreateDirectoryDialog(
    directories: List<DirectoryItemUi>,
    onDismissRequest: () -> Unit,
    onCreateDirectory: (String) -> Unit,
) {
    var directoryName by remember { mutableStateOf("") }
    val trimmedName = directoryName.trim()
    val nameAlreadyExists =
        trimmedName.isNotEmpty() &&
            directories.any { dir ->
                !isSpecialDirectory(dir.id) &&
                    dir.name.trim().equals(trimmedName, ignoreCase = true)
            }
    universalBasicAlertDialog(
        onDismissRequest = onDismissRequest,
        slots =
            UniversalBasicAlertDialogSlots(
                icon = Icons.Rounded.Folder,
                iconContainerColor = MaterialTheme.colorScheme.surfaceContainer,
                iconTintColor = MaterialTheme.colorScheme.onSurfaceVariant,
                title = {
                    Text(
                        text = "Create Directory",
                        fontWeight = FontWeight.W400,
                    )
                },
                input = {
                    directoryOutlinedTextField(
                        modifier = Modifier.padding(top = 5.dp),
                        value = directoryName,
                        onValueChange = { directoryName = it },
                        placeholderText = "Enter directory name...",
                        isError = nameAlreadyExists,
                        errorMessage = if (nameAlreadyExists) DIRECTORY_NAME_TAKEN_ERROR else null,
                        requestInitialFocus = true,
                    )
                },
                actions = {
                    TextButton(
                        onClick = onDismissRequest,
                        contentPadding = PaddingValues(horizontal = 12.dp),
                    ) {
                        Text("Cancel")
                    }

                    Spacer(modifier = Modifier.width(4.dp))

                    Button(
                        onClick = {
                            onCreateDirectory(directoryName)
                            onDismissRequest()
                        },
                        enabled = trimmedName.isNotEmpty() && !nameAlreadyExists,
                        contentPadding = PaddingValues(horizontal = 16.dp),
                        shape = MaterialTheme.shapes.medium,
                    ) {
                        Text("Create")
                    }
                },
            ),
    )
}

internal data class UniversalBasicAlertDialogSlots(
    val icon: ImageVector,
    val iconContainerColor: Color,
    val iconTintColor: Color,
    val title: @Composable () -> Unit,
    val input: @Composable () -> Unit,
    val actions: @Composable RowScope.() -> Unit,
)

@OptIn(ExperimentalMaterial3Api::class)
@Composable
internal fun universalBasicAlertDialog(
    onDismissRequest: () -> Unit,
    modifier: Modifier = Modifier,
    properties: DialogProperties = DialogProperties(usePlatformDefaultWidth = false),
    slots: UniversalBasicAlertDialogSlots,
) {
    BasicAlertDialog(
        onDismissRequest = onDismissRequest,
        modifier =
            modifier
                .fillMaxWidth(0.87f)
                .sizeIn(maxWidth = 560.dp),
        properties = properties,
    ) {
        val dialogFocusManager = LocalFocusManager.current
        Surface(
            shape = MaterialTheme.shapes.extraLarge,
        ) {
            Box {
                Box(
                    modifier =
                        Modifier
                            .matchParentSize()
                            .clickable(
                                interactionSource = remember { MutableInteractionSource() },
                                indication = null,
                            ) {
                                dialogFocusManager.clearFocus(force = true)
                            },
                )
                Column(
                    modifier = Modifier.padding(20.dp),
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.SpaceBetween,
                ) {
                    Box(
                        Modifier
                            .clip(MaterialTheme.shapes.medium)
                            .background(slots.iconContainerColor),
                    ) {
                        Icon(
                            slots.icon,
                            modifier =
                                Modifier
                                    .padding(all = 14.dp)
                                    .size(30.dp),
                            contentDescription = null,
                            tint = slots.iconTintColor,
                        )
                    }
                    Spacer(Modifier.height(10.dp))
                    CompositionLocalProvider(
                        LocalContentColor provides MaterialTheme.colorScheme.onSurface,
                    ) {
                        ProvideTextStyle(MaterialTheme.typography.headlineMedium) {
                            slots.title()
                        }
                    }
                    Spacer(Modifier.height(5.dp))
                    slots.input()
                    Spacer(Modifier.height(10.dp))
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.End,
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        slots.actions(this)
                    }
                }
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun directoriesTopBar(
    showSignOut: Boolean,
    onSignOut: () -> Unit,
    showReturnToSignIn: Boolean,
    onReturnToSignIn: () -> Unit,
    onAddDirectoryClick: () -> Unit,
) {
    val colors = MaterialTheme.colorScheme
    val showAccountAction = showSignOut || showReturnToSignIn
    CenterAlignedTopAppBar(
        title = { Text("Directories", color = colors.onSurface) },
        actions = {
            if (showAccountAction) {
                IconButton(
                    onClick = if (showSignOut) onSignOut else onReturnToSignIn,
                    modifier =
                        if (showSignOut) {
                            Modifier.onboardingTarget(OnboardingTargets.DIRECTORIES_SIGN_OUT)
                        } else {
                            Modifier
                        },
                ) {
                    Icon(
                        imageVector =
                            if (showSignOut) {
                                Icons.AutoMirrored.Rounded.Logout
                            } else {
                                Icons.AutoMirrored.Rounded.Login
                            },
                        contentDescription = if (showSignOut) "Sign out" else "Sign in",
                        tint = colors.onSurface,
                    )
                }
            }
            IconButton(
                onClick = onAddDirectoryClick,
                modifier = Modifier.onboardingTarget(OnboardingTargets.DIRECTORIES_ADD),
            ) {
                Icon(
                    Icons.Rounded.Add,
                    contentDescription = null,
                    tint = colors.onSurface,
                )
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

private fun Modifier.clearFocusOnTap(focusManager: FocusManager): Modifier =
    pointerInput(Unit) {
        detectTapGestures(
            onTap = { focusManager.clearFocus(force = true) },
        )
    }

@Composable
private fun directoriesList(
    directories: List<DirectoryItemUi>,
    searchQuery: String,
    onSearchQueryChange: (String) -> Unit,
    onDirectoryLongClick: (DirectoryItemUi) -> Unit,
    onDirectoryRename: (DirectoryItemUi, String) -> Unit,
    onDirectoryClick: (DirectoryItemUi) -> Unit,
    modifier: Modifier = Modifier,
) {
    var pendingDelete by remember { mutableStateOf<DirectoryItemUi?>(null) }
    var pendingRename by remember { mutableStateOf<DirectoryItemUi?>(null) }
    val focusManager = LocalFocusManager.current

    BackHandler(enabled = pendingRename != null || pendingDelete != null) {
        when {
            pendingRename != null -> pendingRename = null
            pendingDelete != null -> pendingDelete = null
        }
    }

    val allNotesDirectory = remember(directories) { directories.firstOrNull { it.id == ALL_DIRECTORY_ID } }
    val favoritesDirectory =
        remember(directories) { directories.firstOrNull { it.id == FAVORITES_DIRECTORY_ID } }
    val regularDirectories =
        remember(directories) {
            directories.filter { it.id != ALL_DIRECTORY_ID && it.id != FAVORITES_DIRECTORY_ID }
        }
    val totalNotesCount =
        allNotesDirectory?.noteCount ?: directories.sumOf { it.noteCount }
    val recentDirectory =
        remember(totalNotesCount) {
            DirectoryItemUi(id = RECENT_DIRECTORY_ID, name = "Recent", noteCount = totalNotesCount)
        }
    val isSearchActive = searchQuery.isNotBlank()

    Column(
        modifier = modifier.fillMaxSize().clearFocusOnTap(focusManager).padding(horizontal = 12.dp),
    ) {
        directorySearchBar(
            query = searchQuery,
            onQueryChange = onSearchQueryChange,
            modifier = Modifier.onboardingTarget(OnboardingTargets.DIRECTORIES_SEARCH),
        )
        BoxWithConstraints(modifier = Modifier.weight(1f).fillMaxWidth()) {
            LazyColumn(
                modifier =
                    Modifier
                        .fillMaxSize()
                        .heightIn(min = maxHeight + 1.dp),
                contentPadding = PaddingValues(bottom = 12.dp),
            ) {
                fun LazyListScope.addSection(
                    title: String,
                    dirs: List<DirectoryItemUi>,
                    tourHighlightDirectoryId: String? = null,
                ) {
                    if (dirs.isEmpty()) return
                    item { sectionTitle(title = title) }
                    item {
                        directoriesBlock(
                            directories = dirs,
                            onDirectoryClick = onDirectoryClick,
                            onDirectoryLongClick = { pendingDelete = it },
                            tourHighlightDirectoryId = tourHighlightDirectoryId,
                        )
                    }
                }

                item {
                    directoriesHeroPanel(
                        directoriesCount = regularDirectories.size,
                        totalNotesCount = totalNotesCount,
                    )
                }
                allNotesDirectory?.let { allNotes ->
                    addSection("Everything", listOf(allNotes))
                }
                favoritesDirectory?.let { favorites ->
                    addSection("Favorite notes", listOf(favorites))
                }
                if (!isSearchActive) {
                    addSection("Continue working", listOf(recentDirectory))
                }
                addSection(
                    title = "Regular directories",
                    dirs = regularDirectories,
                    tourHighlightDirectoryId =
                        regularDirectories.firstOrNull()?.id ?: allNotesDirectory?.id,
                )

                when {
                    isSearchActive && directories.isEmpty() -> {
                        item {
                            Box(
                                modifier =
                                    Modifier
                                        .fillMaxWidth()
                                        .padding(top = 80.dp)
                                        .heightIn(min = 220.dp),
                                contentAlignment = Alignment.Center,
                            ) {
                                directoriesSearchEmptyState()
                            }
                        }
                    }
                    !isSearchActive && regularDirectories.isEmpty() -> {
                        item {
                            Box(
                                modifier =
                                    Modifier
                                        .fillMaxWidth()
                                        .padding(top = 80.dp)
                                        .heightIn(min = 220.dp),
                                contentAlignment = Alignment.Center,
                            ) {
                                directoriesEmptyState()
                            }
                        }
                    }
                }
            }
        }
    }

    pendingDelete?.let { dir ->
        directoryActionsDialog(
            directory = dir,
            onDelete = {
                onDirectoryLongClick(dir)
                pendingDelete = null
            },
            onRename = {
                pendingDelete = null
                pendingRename = dir
            },
            onDismiss = { pendingDelete = null },
        )
    }
    pendingRename?.let { dir ->
        directoryRenameDialog(
            directories = directories,
            directory = dir,
            onSave = { newName ->
                onDirectoryRename(dir, newName)
                pendingRename = null
            },
            onDismiss = { pendingRename = null },
        )
    }
}

@Composable
private fun directoriesHeroPanel(
    directoriesCount: Int,
    totalNotesCount: Int,
) {
    val colors = MaterialTheme.colorScheme
    Surface(
        color = colors.surfaceContainer,
        shape = MaterialTheme.shapes.large,
        modifier = Modifier.padding(top = 10.dp, bottom = 6.dp),
    ) {
        Row(
            modifier =
                Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 24.dp, vertical = 14.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Icon(
                imageVector = Icons.Rounded.FolderCopy,
                contentDescription = null,
                tint = colors.primary,
                modifier = Modifier.size(25.dp),
            )
            Spacer(Modifier.width(12.dp))
            Column {
                Text(
                    text = "Workspace",
                    style = MaterialTheme.typography.titleMedium,
                    color = colors.onSurface,
                )
                Text(
                    text = "$totalNotesCount notes • $directoriesCount directories",
                    style = MaterialTheme.typography.bodyMedium,
                    color = colors.onSurfaceVariant,
                )
            }
        }
    }
}

@Composable
private fun sectionTitle(title: String) {
    Text(
        text = title,
        style = MaterialTheme.typography.titleSmall,
        color = MaterialTheme.colorScheme.onSurfaceVariant,
        modifier = Modifier.padding(start = 8.dp, top = 10.dp, bottom = 6.dp),
    )
}

@Composable
private fun directoriesBlock(
    directories: List<DirectoryItemUi>,
    onDirectoryClick: (DirectoryItemUi) -> Unit,
    onDirectoryLongClick: (DirectoryItemUi) -> Unit,
    tourHighlightDirectoryId: String? = null,
) {
    Surface(
        color = MaterialTheme.colorScheme.surfaceContainer,
        shape = MaterialTheme.shapes.large,
    ) {
        Column(
            modifier = Modifier.fillMaxWidth(),
        ) {
            directories.forEachIndexed { index, dir ->
                directoryRow(
                    directory = dir,
                    onClick = {
                        onDirectoryClick(dir)
                    },
                    onLongClick = {
                        if (!isSpecialDirectory(dir.id)) {
                            onDirectoryLongClick(dir)
                        }
                    },
                    modifier =
                        Modifier
                            .padding(horizontal = 12.dp, vertical = 0.dp)
                            .then(
                                if (dir.id == tourHighlightDirectoryId) {
                                    Modifier.onboardingTarget(OnboardingTargets.DIRECTORIES_FOLDER_ROW)
                                } else {
                                    Modifier
                                },
                            ),
                )
                if (index < directories.lastIndex) {
                    directoriesListDivider()
                }
            }
        }
    }
}

@Composable
fun directorySearchBar(
    query: String,
    onQueryChange: (String) -> Unit,
    modifier: Modifier = Modifier,
) {
    appSearchField(
        value = query,
        onValueChange = onQueryChange,
        modifier = modifier,
        placeholderText = "Search directories",
    )
}

private fun isSpecialDirectory(directoryId: String): Boolean = isVirtualDirectory(directoryId)

@OptIn(ExperimentalFoundationApi::class)
@Composable
private fun directoryRow(
    directory: DirectoryItemUi,
    onClick: () -> Unit,
    onLongClick: () -> Unit,
    modifier: Modifier = Modifier,
) {
    val colors = MaterialTheme.colorScheme
    val isAllNotes = directory.id == ALL_DIRECTORY_ID
    val isFavorites = directory.id == FAVORITES_DIRECTORY_ID
    Row(
        modifier =
            modifier
                .fillMaxWidth()
                .clip(MaterialTheme.shapes.medium)
                .combinedClickable(
                    onClick = onClick,
                    onLongClick = onLongClick,
                ).padding(horizontal = 12.dp, vertical = 11.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Icon(
            imageVector =
                when {
                    directory.id == RECENT_DIRECTORY_ID -> Icons.Rounded.Schedule
                    isFavorites -> Icons.Rounded.Star
                    isAllNotes -> Icons.Rounded.AllInbox
                    else -> Icons.Rounded.Folder
                },
            contentDescription = null,
            tint =
                if (isAllNotes || isFavorites) {
                    colors.primary
                } else {
                    colors.onSurfaceVariant
                },
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
        Surface(
            color = colors.surfaceVariant,
            shape = CircleShape,
        ) {
            Text(
                text = directory.noteCount.toString(),
                color = colors.onSurfaceVariant,
                modifier = Modifier.padding(horizontal = 8.dp, vertical = 2.dp),
                style = MaterialTheme.typography.labelSmall,
            )
        }
        Spacer(Modifier.width(5.dp))
        Icon(
            Icons.Rounded.ChevronRight,
            contentDescription = null,
            tint = colors.onSurfaceVariant,
        )
    }
}

@Composable
private fun directoriesListDivider() {
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
private fun directoryActionsDialog(
    directory: DirectoryItemUi,
    onDelete: () -> Unit,
    onRename: () -> Unit,
    onDismiss: () -> Unit,
) {
    universalBasicAlertDialog(
        onDismissRequest = onDismiss,
        slots =
            UniversalBasicAlertDialogSlots(
                icon = Icons.Rounded.Edit,
                iconContainerColor = MaterialTheme.colorScheme.surfaceContainer,
                iconTintColor = MaterialTheme.colorScheme.onSurfaceVariant,
                title = {
                    Text("Directory actions")
                },
                input = {
                    Text(
                        text = "Choose action for \"${directory.name}\"",
                        style = MaterialTheme.typography.bodyLarge,
                        textAlign = TextAlign.Center,
                        maxLines = 2,
                        overflow = TextOverflow.Ellipsis,
                    )
                },
                actions = {
                    TextButton(onClick = onRename) {
                        Text("Rename")
                    }

                    Spacer(modifier = Modifier.width(4.dp))

                    Button(
                        onClick = onDelete,
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
private fun directoryRenameDialog(
    directories: List<DirectoryItemUi>,
    directory: DirectoryItemUi,
    onSave: (String) -> Unit,
    onDismiss: () -> Unit,
) {
    var renameName by remember(directory.id) { mutableStateOf(directory.name.toSingleLineText()) }
    val trimmedName = renameName.trim()
    val nameAlreadyExists =
        trimmedName.isNotEmpty() &&
            directories.any { dir ->
                !isSpecialDirectory(dir.id) &&
                    dir.id != directory.id &&
                    dir.name.trim().equals(trimmedName, ignoreCase = true)
            }
    universalBasicAlertDialog(
        onDismissRequest = onDismiss,
        slots =
            UniversalBasicAlertDialogSlots(
                icon = Icons.Rounded.Edit,
                iconContainerColor = MaterialTheme.colorScheme.surfaceContainer,
                iconTintColor = MaterialTheme.colorScheme.onSurfaceVariant,
                title = {
                    Text("Rename directory")
                },
                input = {
                    directoryOutlinedTextField(
                        modifier = Modifier.padding(top = 5.dp),
                        value = renameName,
                        onValueChange = { renameName = it },
                        placeholderText = "Enter directory name...",
                        isError = nameAlreadyExists,
                        errorMessage = if (nameAlreadyExists) DIRECTORY_NAME_TAKEN_ERROR else null,
                        requestInitialFocus = true,
                    )
                },
                actions = {
                    TextButton(onClick = onDismiss) {
                        Text("Cancel")
                    }

                    Spacer(modifier = Modifier.width(4.dp))

                    Button(
                        onClick = { onSave(renameName) },
                        enabled = trimmedName.isNotEmpty() && renameName != directory.name && !nameAlreadyExists,
                        contentPadding = PaddingValues(horizontal = 16.dp),
                        shape = MaterialTheme.shapes.medium,
                    ) {
                        Text("Save")
                    }
                },
            ),
    )
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun directoryOutlinedTextField(
    value: String,
    onValueChange: (String) -> Unit,
    placeholderText: String,
    modifier: Modifier = Modifier,
    enabled: Boolean = true,
    isError: Boolean = false,
    errorMessage: String? = null,
    requestInitialFocus: Boolean = false,
) {
    val focusRequester = remember { FocusRequester() }
    LaunchedEffect(requestInitialFocus) {
        if (requestInitialFocus) {
            focusRequester.requestFocus()
        }
    }
    val interactionSource = remember { MutableInteractionSource() }
    val scheme = MaterialTheme.colorScheme
    val shape = MaterialTheme.shapes.medium
    val colors =
        OutlinedTextFieldDefaults.colors(
            focusedContainerColor = scheme.surfaceContainer,
            unfocusedContainerColor = scheme.surfaceContainer,
            disabledContainerColor = Color.Transparent,
            focusedTextColor = scheme.onSurfaceVariant,
            unfocusedTextColor = scheme.onSurfaceVariant,
            disabledTextColor = scheme.onSurfaceVariant.copy(alpha = 0.38f),
            focusedBorderColor = scheme.outline,
            unfocusedBorderColor = scheme.outline.copy(alpha = 0.5f),
            disabledBorderColor = scheme.outline.copy(alpha = 0.5f),
            cursorColor = scheme.primary,
            errorBorderColor = scheme.error,
            errorCursorColor = scheme.error,
        )
    val textStyle = MaterialTheme.typography.bodyLarge.copy(color = scheme.onSurfaceVariant)
    val contentPadding = PaddingValues(horizontal = 12.dp, vertical = 6.dp)

    BasicTextField(
        value = value,
        onValueChange = { newValue ->
            onValueChange(newValue.toSingleLineText().coerceDirectoryNameLength())
        },
        modifier =
            modifier
                .fillMaxWidth()
                .then(
                    if (requestInitialFocus) {
                        Modifier.focusRequester(focusRequester)
                    } else {
                        Modifier
                    },
                ),
        enabled = enabled,
        textStyle = textStyle,
        singleLine = true,
        maxLines = 1,
        keyboardOptions =
            KeyboardOptions(
                capitalization = KeyboardCapitalization.Words,
                imeAction = ImeAction.Done,
            ),
        cursorBrush = SolidColor(scheme.primary),
        interactionSource = interactionSource,
        decorationBox = { innerTextField ->
            OutlinedTextFieldDefaults.DecorationBox(
                value = value,
                innerTextField = innerTextField,
                enabled = enabled,
                singleLine = true,
                visualTransformation = VisualTransformation.None,
                interactionSource = interactionSource,
                isError = isError,
                supportingText =
                    if (isError && errorMessage != null) {
                        {
                            Text(
                                text = errorMessage,
                                style = MaterialTheme.typography.bodySmall,
                                color = scheme.error,
                            )
                        }
                    } else {
                        null
                    },
                placeholder = {
                    Text(
                        text = placeholderText,
                        style = MaterialTheme.typography.bodyLarge,
                        color = scheme.onSurfaceVariant.copy(alpha = 0.5f),
                    )
                },
                leadingIcon = {
                    Icon(
                        imageVector = Icons.Rounded.TextFields,
                        contentDescription = null,
                        tint = scheme.onSurfaceVariant,
                        modifier = Modifier.size(22.dp),
                    )
                },
                colors = colors,
                contentPadding = contentPadding,
                container = {
                    OutlinedTextFieldDefaults.Container(
                        enabled = enabled,
                        isError = isError,
                        interactionSource = interactionSource,
                        colors = colors,
                        shape = shape,
                    )
                },
            )
        },
    )
}

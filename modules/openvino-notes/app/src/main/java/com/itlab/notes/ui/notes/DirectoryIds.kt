package com.itlab.notes.ui.notes

internal const val ALL_DIRECTORY_ID = "all"
internal const val RECENT_DIRECTORY_ID = "recent"
internal const val FAVORITES_DIRECTORY_ID = "favorites"

/** Max stored/displayed length for user-created folder names. */
internal const val DIRECTORY_NAME_MAX_LENGTH = 40

internal fun String.coerceDirectoryNameLength(): String = take(DIRECTORY_NAME_MAX_LENGTH)

internal fun isVirtualDirectory(directoryId: String): Boolean =
    directoryId == ALL_DIRECTORY_ID ||
        directoryId == RECENT_DIRECTORY_ID ||
        directoryId == FAVORITES_DIRECTORY_ID

internal fun canCreateNotesInDirectory(directoryId: String): Boolean = !isVirtualDirectory(directoryId)

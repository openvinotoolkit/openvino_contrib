package com.itlab.notes.onboarding

import com.itlab.notes.ui.NotesUiScreen

enum class OnboardingScreenKind {
    Directories,
    DirectoryNotes,
}

data class OnboardingTourStep(
    val targetKey: String?,
    val title: String,
    val description: String,
    val requiredScreen: OnboardingScreenKind? = null,
    val requiresSignIn: Boolean = false,
)

fun NotesUiScreen.onboardingScreenKind(): OnboardingScreenKind? =
    when (this) {
        NotesUiScreen.Directories -> OnboardingScreenKind.Directories
        is NotesUiScreen.DirectoryNotes -> OnboardingScreenKind.DirectoryNotes
        is NotesUiScreen.NoteEditor -> null
    }

object OnboardingTargets {
    const val DIRECTORIES_SEARCH = "directories_search"
    const val DIRECTORIES_ADD = "directories_add"
    const val DIRECTORIES_FOLDER_ROW = "directories_folder_row"
    const val DIRECTORIES_SIGN_OUT = "directories_sign_out"
    const val NOTES_FAB = "notes_fab"
    const val NOTES_SEARCH = "notes_search"
    const val NOTES_NOTE_ROW = "notes_note_row"
}

fun buildOnboardingTourSteps(showSignOut: Boolean): List<OnboardingTourStep> =
    buildList {
        add(
            OnboardingTourStep(
                targetKey = OnboardingTargets.DIRECTORIES_SEARCH,
                title = "Search",
                description = "Find directories quickly. Inside a folder you can search notes too.",
                requiredScreen = OnboardingScreenKind.Directories,
            ),
        )
        add(
            OnboardingTourStep(
                targetKey = OnboardingTargets.DIRECTORIES_ADD,
                title = "New directory",
                description = "Tap + to create a folder for your notes.",
                requiredScreen = OnboardingScreenKind.Directories,
            ),
        )
        add(
            OnboardingTourStep(
                targetKey = OnboardingTargets.DIRECTORIES_FOLDER_ROW,
                title = "Folders",
                description = "Tap to open. Long-press a custom folder to rename or delete it in a dialog.",
                requiredScreen = OnboardingScreenKind.Directories,
            ),
        )
        if (showSignOut) {
            add(
                OnboardingTourStep(
                    targetKey = OnboardingTargets.DIRECTORIES_SIGN_OUT,
                    title = "Sign out",
                    description =
                        "Use this when switching accounts. Local notes from the " +
                            "session are cleared on sign out.",
                    requiredScreen = OnboardingScreenKind.Directories,
                    requiresSignIn = true,
                ),
            )
        }
        add(
            OnboardingTourStep(
                targetKey = null,
                title = "Open a folder",
                description = "Tap any directory to continue the tour and see note actions.",
                requiredScreen = OnboardingScreenKind.DirectoryNotes,
            ),
        )
        add(
            OnboardingTourStep(
                targetKey = OnboardingTargets.NOTES_FAB,
                title = "New note",
                description = "Tap + to create a note in this folder.",
                requiredScreen = OnboardingScreenKind.DirectoryNotes,
            ),
        )
        add(
            OnboardingTourStep(
                targetKey = OnboardingTargets.NOTES_SEARCH,
                title = "Search notes",
                description = "Filter notes by title or content in the current folder.",
                requiredScreen = OnboardingScreenKind.DirectoryNotes,
            ),
        )
        add(
            OnboardingTourStep(
                targetKey = OnboardingTargets.NOTES_NOTE_ROW,
                title = "Notes list",
                description =
                    "Tap a note to edit. Long-press to select several notes, then move or delete" +
                        " them from the top bar.",
                requiredScreen = OnboardingScreenKind.DirectoryNotes,
            ),
        )
    }

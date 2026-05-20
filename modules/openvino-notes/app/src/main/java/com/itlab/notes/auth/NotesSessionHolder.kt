package com.itlab.notes.auth

/**
 * Tracks whether the UI is in offline mode so [com.itlab.domain.repository.AuthRepository]
 * resolves the correct Room userId (Firebase uid vs local offline profile).
 */
class NotesSessionHolder {
    @Volatile
    var continueOffline: Boolean = false

    fun resolveUserId(firebaseUserId: String?): String? =
        if (continueOffline) {
            NotesUserIds.LOCAL_OFFLINE
        } else {
            firebaseUserId
        }
}

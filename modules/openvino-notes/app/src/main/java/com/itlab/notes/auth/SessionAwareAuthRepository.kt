package com.itlab.notes.auth

import com.itlab.data.cloud.AuthManager
import com.itlab.domain.repository.AuthRepository

class SessionAwareAuthRepository(
    private val authManager: AuthManager,
    private val sessionHolder: NotesSessionHolder,
) : AuthRepository {
    override fun getCurrentUserId(): String? = sessionHolder.resolveUserId(authManager.getCurrentUserId())
}

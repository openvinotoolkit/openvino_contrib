package com.itlab.notes.ui.auth

import android.app.Application
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.google.android.gms.auth.api.signin.GoogleSignIn
import com.google.android.gms.auth.api.signin.GoogleSignInOptions
import com.google.firebase.auth.FirebaseAuth
import com.google.firebase.auth.FirebaseAuthInvalidCredentialsException
import com.google.firebase.auth.FirebaseAuthInvalidUserException
import com.google.firebase.auth.FirebaseAuthUserCollisionException
import com.google.firebase.auth.FirebaseAuthWeakPasswordException
import com.google.firebase.auth.GoogleAuthProvider
import com.itlab.domain.cloud.SyncCheckpointStore
import com.itlab.notes.R
import com.itlab.notes.auth.AppSessionPreferences
import com.itlab.notes.auth.NotesSessionHolder
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.tasks.await

enum class AuthScreenStep {
    ChooseMethod,
    Email,
}

data class AuthUiState(
    val step: AuthScreenStep = AuthScreenStep.ChooseMethod,
    val isSessionReady: Boolean = false,
    /** User may enter the notes app (explicit sign-in or restored Firebase session). */
    val isSessionActive: Boolean = false,
    val continueOffline: Boolean = false,
    val isLoading: Boolean = false,
    val isSignUpMode: Boolean = false,
    val errorMessage: String? = null,
    val successMessage: String? = null,
)

class AuthViewModel(
    private val firebaseAuth: FirebaseAuth,
    private val app: Application,
    private val appSessionPreferences: AppSessionPreferences,
    private val syncCheckpointStore: SyncCheckpointStore,
    private val notesSessionHolder: NotesSessionHolder,
) : ViewModel() {
    private var shouldActivateSession = firebaseAuth.currentUser != null

    private val _uiState =
        MutableStateFlow(
            AuthUiState(
                isSessionActive = firebaseAuth.currentUser != null && shouldActivateSession,
            ),
        )
    val uiState: StateFlow<AuthUiState> = _uiState.asStateFlow()

    val sessionKey: String?
        get() =
            when {
                _uiState.value.continueOffline -> OFFLINE_SESSION_KEY
                _uiState.value.isSessionActive -> firebaseAuth.currentUser?.uid
                else -> null
            }

    private val authStateListener =
        FirebaseAuth.AuthStateListener { auth ->
            val signedIn = auth.currentUser != null
            val sessionActive = signedIn && shouldActivateSession
            _uiState.update {
                it.copy(
                    isSessionActive = sessionActive,
                    continueOffline = if (sessionActive) false else it.continueOffline,
                    isLoading = false,
                )
            }
            if (sessionActive) {
                viewModelScope.launch { appSessionPreferences.setContinueOffline(false) }
            }
        }

    init {
        firebaseAuth.addAuthStateListener(authStateListener)
        viewModelScope.launch {
            appSessionPreferences.continueOffline.collect { offline ->
                _uiState.update { state ->
                    val sessionActive = firebaseAuth.currentUser != null && shouldActivateSession
                    state.copy(
                        continueOffline = if (sessionActive) false else offline,
                        isSessionReady = true,
                    )
                }
            }
        }
        viewModelScope.launch {
            uiState.collect { syncNotesSessionHolder(it) }
        }
    }

    private fun syncNotesSessionHolder(state: AuthUiState) {
        notesSessionHolder.continueOffline = state.continueOffline && !state.isSessionActive
    }

    override fun onCleared() {
        firebaseAuth.removeAuthStateListener(authStateListener)
        super.onCleared()
    }

    fun openEmailStep() {
        _uiState.update {
            it.copy(
                step = AuthScreenStep.Email,
                isSignUpMode = false,
                errorMessage = null,
                successMessage = null,
            )
        }
    }

    fun backToMethodChoice() {
        _uiState.update {
            it.copy(
                step = AuthScreenStep.ChooseMethod,
                isSignUpMode = false,
                errorMessage = null,
                successMessage = null,
                isLoading = false,
            )
        }
    }

    fun switchToSignUpMode() {
        _uiState.update {
            it.copy(
                isSignUpMode = true,
                errorMessage = null,
                successMessage = null,
            )
        }
    }

    fun switchToSignInMode() {
        _uiState.update {
            it.copy(
                isSignUpMode = false,
                errorMessage = null,
                successMessage = null,
            )
        }
    }

    fun backFromEmailStep() {
        if (_uiState.value.isSignUpMode) {
            switchToSignInMode()
        } else {
            backToMethodChoice()
        }
    }

    fun continueOffline() {
        _uiState.update { it.copy(continueOffline = true, errorMessage = null) }
        viewModelScope.launch { appSessionPreferences.setContinueOffline(true) }
    }

    private suspend fun clearOfflineSession() {
        appSessionPreferences.setContinueOffline(false)
    }

    /** Leaves offline mode and returns to the sign-in choice screen. Offline notes stay in Room. */
    fun exitOfflineToSignIn() {
        viewModelScope.launch {
            clearOfflineSession()
            _uiState.update {
                it.copy(
                    step = AuthScreenStep.ChooseMethod,
                    continueOffline = false,
                    isSessionActive = false,
                    isLoading = false,
                    errorMessage = null,
                    successMessage = null,
                )
            }
        }
    }

    fun clearError() {
        _uiState.update { it.copy(errorMessage = null) }
    }

    fun clearSuccess() {
        _uiState.update { it.copy(successMessage = null) }
    }

    fun reportError(message: String) {
        _uiState.update { it.copy(isLoading = false, errorMessage = message) }
    }

    fun signInWithEmail(
        email: String,
        password: String,
    ) {
        val trimmedEmail = email.trim()
        if (trimmedEmail.isEmpty() || password.isEmpty()) {
            _uiState.update { it.copy(errorMessage = "Email and password are required.") }
            return
        }
        viewModelScope.launch {
            _uiState.update { it.copy(isLoading = true, errorMessage = null, successMessage = null) }
            shouldActivateSession = true
            runCatching {
                firebaseAuth.signInWithEmailAndPassword(trimmedEmail, password).await()
            }.onFailure { error ->
                shouldActivateSession = false
                _uiState.update {
                    it.copy(
                        isLoading = false,
                        errorMessage = mapAuthError(error),
                    )
                }
            }
        }
    }

    fun signUpWithEmail(
        email: String,
        password: String,
    ) {
        val trimmedEmail = email.trim()
        if (trimmedEmail.isEmpty() || password.isEmpty()) {
            _uiState.update { it.copy(errorMessage = "Email and password are required.") }
            return
        }
        viewModelScope.launch {
            _uiState.update { it.copy(isLoading = true, errorMessage = null, successMessage = null) }
            shouldActivateSession = false
            runCatching {
                firebaseAuth.createUserWithEmailAndPassword(trimmedEmail, password).await()
                firebaseAuth.signOut()
            }.onSuccess {
                _uiState.update {
                    it.copy(
                        isLoading = false,
                        isSessionActive = false,
                        successMessage =
                            "Account created. Switch to Sign in below when you are ready.",
                        errorMessage = null,
                    )
                }
            }.onFailure { error ->
                _uiState.update {
                    it.copy(
                        isLoading = false,
                        errorMessage = mapAuthError(error),
                    )
                }
            }
        }
    }

    fun signInWithGoogle(idToken: String) {
        viewModelScope.launch {
            _uiState.update { it.copy(isLoading = true, errorMessage = null, successMessage = null) }
            shouldActivateSession = true
            runCatching {
                val credential = GoogleAuthProvider.getCredential(idToken, null)
                firebaseAuth.signInWithCredential(credential).await()
            }.onFailure { error ->
                shouldActivateSession = false
                _uiState.update {
                    it.copy(
                        isLoading = false,
                        errorMessage = mapAuthError(error),
                    )
                }
            }
        }
    }

    fun signOut() {
        viewModelScope.launch {
            _uiState.update { it.copy(isLoading = true, errorMessage = null, successMessage = null) }
            shouldActivateSession = false
            runCatching {
                // Keep notes in Room (keyed by userId). They reappear after sign-in; cloud sync can merge.
                firebaseAuth.currentUser?.uid?.let { uid ->
                    syncCheckpointStore.clearUser(uid)
                }
                clearOfflineSession()
                firebaseAuth.signOut()
                signOutGoogle()
            }.onFailure { error ->
                _uiState.update {
                    it.copy(
                        isLoading = false,
                        errorMessage = error.message ?: "Sign out failed.",
                    )
                }
            }
            _uiState.update {
                it.copy(
                    step = AuthScreenStep.ChooseMethod,
                    continueOffline = false,
                    isLoading = false,
                    isSessionActive = false,
                )
            }
        }
    }

    private suspend fun signOutGoogle() {
        val webClientId = app.getString(R.string.default_web_client_id)
        if (webClientId.isBlank()) return
        val options =
            GoogleSignInOptions
                .Builder(GoogleSignInOptions.DEFAULT_SIGN_IN)
                .requestIdToken(webClientId)
                .requestEmail()
                .build()
        GoogleSignIn.getClient(app, options).signOut().await()
    }

    private fun mapAuthError(error: Throwable): String =
        when (error) {
            is FirebaseAuthInvalidUserException ->
                "No account found for this email. Try creating an account."
            is FirebaseAuthInvalidCredentialsException ->
                "Invalid email or password."
            is FirebaseAuthWeakPasswordException ->
                "Password must be at least 6 characters."
            is FirebaseAuthUserCollisionException ->
                "An account with this email already exists. Sign in instead."
            else -> error.message ?: "Authentication failed. Please try again."
        }

    companion object {
        const val OFFLINE_SESSION_KEY = "offline"
    }
}

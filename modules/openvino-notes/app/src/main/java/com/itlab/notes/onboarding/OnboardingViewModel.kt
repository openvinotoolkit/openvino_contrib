package com.itlab.notes.onboarding

import androidx.compose.ui.geometry.Rect
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.itlab.notes.ui.NotesUiScreen
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

private data class TourCoreState(
    val isReady: Boolean,
    val welcomeDone: Boolean,
    val tourDone: Boolean,
    val tourOn: Boolean,
    val step: Int,
)

data class OnboardingUiState(
    val isReady: Boolean = false,
    val showWelcome: Boolean = false,
    val showTour: Boolean = false,
    val tourStepIndex: Int = 0,
    val showSignOutStep: Boolean = false,
    val currentScreenKind: OnboardingScreenKind? = OnboardingScreenKind.Directories,
)

class OnboardingViewModel(
    private val preferences: OnboardingPreferences,
) : ViewModel() {
    private val welcomeCompleted = MutableStateFlow(false)
    private val tourCompleted = MutableStateFlow(false)
    private val tourActive = MutableStateFlow(false)
    private val tourStepIndex = MutableStateFlow(0)
    private val showSignOutStep = MutableStateFlow(false)
    private val currentScreenKind = MutableStateFlow<OnboardingScreenKind?>(OnboardingScreenKind.Directories)
    private val targetBounds = MutableStateFlow<Map<String, Rect>>(emptyMap())
    val targetBoundsState: StateFlow<Map<String, Rect>> = targetBounds.asStateFlow()
    private val preferencesLoaded = MutableStateFlow(false)
    private val pendingTourStart = MutableStateFlow(false)

    val uiState: StateFlow<OnboardingUiState> =
        combine(
            combine(
                preferencesLoaded,
                welcomeCompleted,
                tourCompleted,
                tourActive,
                tourStepIndex,
            ) { loaded, welcomeDone, tourDone, tourOn, step ->
                TourCoreState(
                    isReady = loaded,
                    welcomeDone = welcomeDone,
                    tourDone = tourDone,
                    tourOn = tourOn,
                    step = step,
                )
            },
            showSignOutStep,
            currentScreenKind,
        ) { core, signOut, screen ->
            OnboardingUiState(
                isReady = core.isReady,
                showWelcome = core.isReady && !core.welcomeDone,
                showTour = core.isReady && core.welcomeDone && !core.tourDone && core.tourOn,
                tourStepIndex = core.step,
                showSignOutStep = signOut,
                currentScreenKind = screen,
            )
        }.stateIn(
            scope = viewModelScope,
            started = SharingStarted.WhileSubscribed(5_000),
            initialValue = OnboardingUiState(isReady = false),
        )

    val tourSteps: StateFlow<List<OnboardingTourStep>> =
        showSignOutStep
            .combine(tourStepIndex) { signOut, _ -> buildOnboardingTourSteps(signOut) }
            .stateIn(
                scope = viewModelScope,
                started = SharingStarted.WhileSubscribed(5_000),
                initialValue = buildOnboardingTourSteps(false),
            )

    init {
        viewModelScope.launch {
            preferences.welcomeCompleted.collect { welcomeCompleted.value = it }
        }
        viewModelScope.launch {
            preferences.tourCompleted.collect { tourCompleted.value = it }
        }
        viewModelScope.launch {
            preferences.welcomeCompleted.first()
            preferences.tourCompleted.first()
            preferencesLoaded.value = true
        }
    }

    fun startTourIfNeeded() {
        if (!welcomeCompleted.value || tourCompleted.value) return
        if (tourActive.value) {
            pendingTourStart.value = false
            return
        }
        pendingTourStart.value = true
    }

    /** Call when the main notes UI is visible (after auth / offline). */
    fun activateTourIfPending() {
        if (!pendingTourStart.value || tourCompleted.value || !welcomeCompleted.value) return
        pendingTourStart.value = false
        tourActive.value = true
        tourStepIndex.value = 0
    }

    fun updateShowSignOutStep(show: Boolean) {
        showSignOutStep.value = show
    }

    fun updateCurrentScreen(screen: NotesUiScreen) {
        currentScreenKind.value = screen.onboardingScreenKind()
    }

    fun registerTarget(
        key: String,
        bounds: Rect?,
    ) {
        targetBounds.update { current ->
            if (bounds == null) {
                current - key
            } else {
                current + (key to bounds)
            }
        }
    }

    fun completeWelcome() {
        viewModelScope.launch {
            preferences.setWelcomeCompleted()
            welcomeCompleted.value = true
            pendingTourStart.value = true
        }
    }

    fun skipWelcome() {
        completeWelcome()
    }

    fun skipTour() {
        finishTour()
    }

    fun nextTourStep() {
        val steps = buildOnboardingTourSteps(showSignOutStep.value)
        val clampedIndex = tourStepIndex.value.coerceIn(0, steps.lastIndex)
        if (clampedIndex >= steps.lastIndex) {
            finishTour()
        } else {
            tourStepIndex.value = clampedIndex + 1
        }
    }

    fun previousTourStep() {
        if (tourStepIndex.value > 0) {
            tourStepIndex.value -= 1
        }
    }

    private fun finishTour() {
        viewModelScope.launch {
            preferences.setTourCompleted()
            tourCompleted.value = true
            tourActive.value = false
        }
    }
}

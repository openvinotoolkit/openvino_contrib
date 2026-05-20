package com.itlab

import com.itlab.domain.repository.AuthRepository
import com.itlab.domain.usecase.aiusecase.ReleaseNoteAiUseCase
import com.itlab.domain.usecase.aiusecase.RewriteNoteUseCase
import com.itlab.domain.usecase.aiusecase.SuggestImageTagsUseCase
import com.itlab.domain.usecase.aiusecase.SuggestSummaryUseCase
import com.itlab.domain.usecase.aiusecase.SuggestTagsUseCase
import com.itlab.domain.usecase.aiusecase.WarmUpNoteAiUseCase
import com.itlab.domain.usecase.folderusecase.CreateFolderUseCase
import com.itlab.domain.usecase.folderusecase.DeleteFolderUseCase
import com.itlab.domain.usecase.folderusecase.GetFolderUseCase
import com.itlab.domain.usecase.folderusecase.ObserveFoldersUseCase
import com.itlab.domain.usecase.folderusecase.UpdateFolderUseCase
import com.itlab.domain.usecase.noteusecase.ApplyRewriteUseCase
import com.itlab.domain.usecase.noteusecase.ApplySummaryUseCase
import com.itlab.domain.usecase.noteusecase.ApplyTagsUseCase
import com.itlab.domain.usecase.noteusecase.CreateNoteUseCase
import com.itlab.domain.usecase.noteusecase.DeleteNoteUseCase
import com.itlab.domain.usecase.noteusecase.GetAllFavoritesUseCase
import com.itlab.domain.usecase.noteusecase.GetNoteUseCase
import com.itlab.domain.usecase.noteusecase.GetUserIdUseCase
import com.itlab.domain.usecase.noteusecase.MoveNoteToFolderUseCase
import com.itlab.domain.usecase.noteusecase.ObserveNotesByFolderUseCase
import com.itlab.domain.usecase.noteusecase.ObserveNotesUseCase
import com.itlab.domain.usecase.noteusecase.SearchNotesUseCase
import com.itlab.domain.usecase.noteusecase.SwitchFavoriteUseCase
import com.itlab.domain.usecase.noteusecase.UpdateNoteUseCase
import com.itlab.domain.usecase.noteusecase.ValidateDuplicateNoteTitleUseCase
import com.itlab.notes.auth.AppSessionPreferences
import com.itlab.notes.auth.NotesSessionHolder
import com.itlab.notes.auth.SessionAwareAuthRepository
import com.itlab.notes.onboarding.OnboardingPreferences
import com.itlab.notes.onboarding.OnboardingViewModel
import com.itlab.notes.ui.NotesUseCases
import com.itlab.notes.ui.NotesViewModel
import com.itlab.notes.ui.auth.AuthViewModel
import org.koin.android.ext.koin.androidApplication
import org.koin.core.module.dsl.viewModel
import org.koin.core.module.dsl.viewModelOf
import org.koin.dsl.module

val appModule =
    module {
        single { NotesSessionHolder() }
        single<AuthRepository> { SessionAwareAuthRepository(get(), get()) }
        single { OnboardingPreferences(androidApplication()) }
        single { AppSessionPreferences(androidApplication()) }
        factory { ValidateDuplicateNoteTitleUseCase(get(), get()) }
        factory { CreateNoteUseCase(get(), get()) }
        factory { CreateFolderUseCase(get(), get()) }
        factory { DeleteFolderUseCase(get(), get(), get()) }
        factory { DeleteNoteUseCase(get(), get()) }
        factory { UpdateNoteUseCase(get(), get()) }
        factory { UpdateFolderUseCase(get(), get()) }
        factory { GetFolderUseCase(get(), get()) }
        factory { ObserveNotesByFolderUseCase(get(), get()) }
        factory { ObserveFoldersUseCase(get(), get()) }
        factory { MoveNoteToFolderUseCase(get(), get(), get()) }
        factory { ObserveNotesUseCase(get(), get()) }
        factory { GetUserIdUseCase(get()) }
        factory { SearchNotesUseCase(get(), get()) }
        factory { SwitchFavoriteUseCase(get(), get()) }
        factory { GetAllFavoritesUseCase(get(), get()) }
        factory { GetNoteUseCase(get(), get()) }
        factory { SuggestSummaryUseCase(get(), get(), get()) }
        factory { SuggestTagsUseCase(get(), get(), get()) }
        factory { SuggestImageTagsUseCase(get(), get(), get()) }
        factory { RewriteNoteUseCase(get(), get(), get()) }
        factory { WarmUpNoteAiUseCase(get()) }
        factory { ReleaseNoteAiUseCase(get()) }
        factory { ApplySummaryUseCase(get(), get()) }
        factory { ApplyTagsUseCase(get(), get()) }
        factory { ApplyRewriteUseCase(get(), get()) }
        factory {
            NotesUseCases(
                createFolderUseCase = get(),
                deleteFolderUseCase = get(),
                createNoteUseCase = get(),
                deleteNoteUseCase = get(),
                updateNoteUseCase = get(),
                observeNotesByFolderUseCase = get(),
                observeFoldersUseCase = get(),
                updateFolderUseCase = get(),
                getFolderUseCase = get(),
                moveNoteToFolderUseCase = get(),
                observeNotesUseCase = get(),
                getUserIdUseCase = get(),
                searchNotesUseCase = get(),
                switchFavoriteUseCase = get(),
                getAllFavoritesUseCase = get(),
                getNoteUseCase = get(),
                suggestSummaryUseCase = get(),
                suggestTagsUseCase = get(),
                suggestImageTagsUseCase = get(),
                rewriteNoteUseCase = get(),
                warmUpNoteAiUseCase = get(),
                releaseNoteAiUseCase = get(),
                applySummaryUseCase = get(),
                applyTagsUseCase = get(),
                applyRewriteUseCase = get(),
            )
        }

        viewModel {
            NotesViewModel(
                useCases = get(),
                syncScheduler = get(),
                syncManager = get(),
                syncCheckpointStore = get(),
            )
        }
        viewModelOf(::OnboardingViewModel)
        viewModel {
            AuthViewModel(
                firebaseAuth = get(),
                app = androidApplication(),
                appSessionPreferences = get(),
                syncCheckpointStore = get(),
                notesSessionHolder = get(),
            )
        }
    }

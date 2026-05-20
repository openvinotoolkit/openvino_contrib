package com.itlab.notes.ui

import com.itlab.domain.usecase.folderusecase.CreateFolderUseCase
import com.itlab.domain.usecase.folderusecase.DeleteFolderUseCase
import com.itlab.domain.usecase.folderusecase.GetFolderUseCase
import com.itlab.domain.usecase.folderusecase.ObserveFoldersUseCase
import com.itlab.domain.usecase.folderusecase.UpdateFolderUseCase
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

data class NotesUseCases(
    val createFolderUseCase: CreateFolderUseCase,
    val deleteFolderUseCase: DeleteFolderUseCase,
    val createNoteUseCase: CreateNoteUseCase,
    val deleteNoteUseCase: DeleteNoteUseCase,
    val updateNoteUseCase: UpdateNoteUseCase,
    val observeNotesByFolderUseCase: ObserveNotesByFolderUseCase,
    val observeFoldersUseCase: ObserveFoldersUseCase,
    val updateFolderUseCase: UpdateFolderUseCase,
    val getFolderUseCase: GetFolderUseCase,
    val moveNoteToFolderUseCase: MoveNoteToFolderUseCase,
    val observeNotesUseCase: ObserveNotesUseCase,
    val getUserIdUseCase: GetUserIdUseCase,
    val searchNotesUseCase: SearchNotesUseCase,
    val switchFavoriteUseCase: SwitchFavoriteUseCase,
    val getAllFavoritesUseCase: GetAllFavoritesUseCase,
    val getNoteUseCase: GetNoteUseCase,
)

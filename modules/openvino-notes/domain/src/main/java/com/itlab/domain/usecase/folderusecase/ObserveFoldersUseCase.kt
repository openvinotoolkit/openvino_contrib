package com.itlab.domain.usecase.folderusecase

import com.itlab.domain.model.NoteFolder
import com.itlab.domain.repository.NoteFolderRepository
import com.itlab.domain.usecase.noteusecase.GetUserIdUseCase
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.emptyFlow

class ObserveFoldersUseCase(
    private val repo: NoteFolderRepository,
    private val getUserIdUseCase: GetUserIdUseCase,
) {
    operator fun invoke(): Flow<List<NoteFolder>> {
        val userId = getUserIdUseCase()

        if (userId == null) {
            return emptyFlow()
        }

        return repo.observeFolders(userId)
    }
}

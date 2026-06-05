package com.itlab.domain.usecase.folderusecase

import com.itlab.domain.model.NoteFolder
import com.itlab.domain.repository.NoteFolderRepository
import com.itlab.domain.usecase.noteusecase.GetUserIdUseCase

class GetFolderUseCase(
    private val repo: NoteFolderRepository,
    private val getUserIdUseCase: GetUserIdUseCase,
) {
    suspend operator fun invoke(id: String): NoteFolder? {
        val userId = getUserIdUseCase()

        if (userId == null) {
            return null
        }

        return repo.getFolderById(id, userId)
    }
}

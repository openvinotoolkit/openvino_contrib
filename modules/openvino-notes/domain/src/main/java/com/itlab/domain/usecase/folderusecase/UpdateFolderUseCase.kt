package com.itlab.domain.usecase.folderusecase

import com.itlab.domain.model.NoteFolder
import com.itlab.domain.repository.NoteFolderRepository
import com.itlab.domain.usecase.noteusecase.GetUserIdUseCase
import com.itlab.domain.usecase.requireNotBlank
import kotlinx.coroutines.flow.first
import kotlin.time.Clock

class UpdateFolderUseCase(
    private val repo: NoteFolderRepository,
    private val getUserIdUseCase: GetUserIdUseCase,
) {
    suspend operator fun invoke(folder: NoteFolder): Result<Unit> =
        runCatching {
            val userId = getUserIdUseCase()

            if (userId == null) {
                return Result.failure(IllegalStateException("User must be authenticated"))
            }

            requireNotBlank(folder.id, "Folder id")
            require(folder.id != "all") { "System folder 'all' cannot be renamed" }
            val normalizedName = folder.name.trim()
            requireNotBlank(normalizedName, "Folder name")
            val hasDuplicateName =
                repo.observeFolders(userId).first().any { existing ->
                    existing.id != folder.id && existing.name.trim().equals(normalizedName, ignoreCase = true)
                }
            require(!hasDuplicateName) { "Folder with name '$normalizedName' already exists" }
            val folder = folder.copy(name = normalizedName, updatedAt = Clock.System.now())
            repo.updateFolder(folder)
        }
}

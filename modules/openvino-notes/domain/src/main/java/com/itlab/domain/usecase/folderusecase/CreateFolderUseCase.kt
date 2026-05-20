package com.itlab.domain.usecase.folderusecase

import com.itlab.domain.model.NoteFolder
import com.itlab.domain.repository.NoteFolderRepository
import com.itlab.domain.usecase.noteusecase.GetUserIdUseCase
import com.itlab.domain.usecase.requireNotBlank
import kotlinx.coroutines.flow.first
import java.util.UUID
import kotlin.time.Clock

class CreateFolderUseCase(
    private val repo: NoteFolderRepository,
    private val getUserIdUseCase: GetUserIdUseCase,
) {
    suspend operator fun invoke(folder: NoteFolder): Result<String> =
        runCatching {
            val userId = getUserIdUseCase()

            if (userId == null) {
                return Result.failure(IllegalStateException("User must be authenticated"))
            }

            val normalizedName = folder.name.trim()
            requireNotBlank(normalizedName, "Folder name")
            val hasDuplicateName =
                repo.observeFolders(userId).first().any { existing ->
                    existing.name.trim().equals(normalizedName, ignoreCase = true)
                }
            require(!hasDuplicateName) { "Folder with name '$normalizedName' already exists" }

            val now = Clock.System.now()
            val folder =
                folder.copy(
                    id = UUID.randomUUID().toString(),
                    name = normalizedName,
                    createdAt = now,
                    updatedAt = now,
                )
            repo.createFolder(folder)
        }
}

package com.itlab.domain.usecase.aiusecase

import com.itlab.domain.ai.NoteAiService
import kotlinx.coroutines.CancellationException

class WarmUpNoteAiUseCase(
    private val ai: NoteAiService,
) {
    suspend operator fun invoke(): Result<Unit> =
        runCatching {
            ai.warmUp()
        }.onFailure { error ->
            if (error is CancellationException) throw error
        }
}

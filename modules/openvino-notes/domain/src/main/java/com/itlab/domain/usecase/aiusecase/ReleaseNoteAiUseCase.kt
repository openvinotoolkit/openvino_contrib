package com.itlab.domain.usecase.aiusecase

import com.itlab.domain.ai.NoteAiService

class ReleaseNoteAiUseCase(
    private val ai: NoteAiService,
) {
    operator fun invoke(): Result<Unit> =
        runCatching {
            ai.release()
        }
}

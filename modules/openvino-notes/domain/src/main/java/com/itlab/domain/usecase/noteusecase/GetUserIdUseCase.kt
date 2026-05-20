package com.itlab.domain.usecase.noteusecase

import com.itlab.domain.repository.AuthRepository

class GetUserIdUseCase(
    private val repository: AuthRepository,
) {
    operator fun invoke(): String? = repository.getCurrentUserId()
}

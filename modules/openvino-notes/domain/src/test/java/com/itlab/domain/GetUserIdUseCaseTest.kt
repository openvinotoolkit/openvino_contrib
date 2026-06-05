package com.itlab.domain

import com.itlab.domain.repository.AuthRepository
import com.itlab.domain.usecase.noteusecase.GetUserIdUseCase
import io.mockk.MockKAnnotations
import io.mockk.every
import io.mockk.impl.annotations.MockK
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Before
import org.junit.Test

class GetUserIdUseCaseTest {
    @MockK
    lateinit var authRepository: AuthRepository

    private lateinit var getUserIdUseCase: GetUserIdUseCase

    @Before
    fun setUp() {
        MockKAnnotations.init(this)
        getUserIdUseCase = GetUserIdUseCase(authRepository)
    }

    @Test
    fun `invoke should return user id from repository`() {
        val expectedId = "user_abc_123"
        every { authRepository.getCurrentUserId() } returns expectedId

        val result = getUserIdUseCase()

        assertEquals(expectedId, result)
    }

    @Test
    fun `invoke should return null when user is not authorized`() {
        every { authRepository.getCurrentUserId() } returns null

        val result = getUserIdUseCase()

        assertNull(result)
    }
}

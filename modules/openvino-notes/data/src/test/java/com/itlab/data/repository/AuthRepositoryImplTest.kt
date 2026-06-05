package com.itlab.data.repository

import com.itlab.data.cloud.AuthManager
import io.mockk.MockKAnnotations
import io.mockk.every
import io.mockk.impl.annotations.MockK
import io.mockk.verify
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Before
import org.junit.Test

class AuthRepositoryImplTest {
    @MockK
    lateinit var authManager: AuthManager

    private lateinit var authRepository: AuthRepositoryImpl

    @Before
    fun setUp() {
        MockKAnnotations.init(this)
        authRepository = AuthRepositoryImpl(authManager)
    }

    @Test
    fun `getCurrentUserId should return uid from authManager`() {
        val expectedUid = "user_12345"
        every { authManager.getCurrentUserId() } returns expectedUid

        val result = authRepository.getCurrentUserId()

        assertEquals(expectedUid, result)
        verify(exactly = 1) { authManager.getCurrentUserId() }
    }

    @Test
    fun `getCurrentUserId should return null when authManager returns null`() {
        every { authManager.getCurrentUserId() } returns null

        val result = authRepository.getCurrentUserId()

        assertNull(result)
        verify(exactly = 1) { authManager.getCurrentUserId() }
    }
}

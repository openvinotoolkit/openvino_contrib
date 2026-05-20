package com.itlab.data.cloud

import android.content.Context
import android.content.Intent
import androidx.test.core.app.ApplicationProvider
import com.firebase.ui.auth.AuthUI
import com.google.android.gms.tasks.Tasks
import com.google.firebase.FirebaseApp
import com.google.firebase.FirebaseOptions
import com.google.firebase.auth.FirebaseAuth
import com.google.firebase.auth.FirebaseUser
import io.mockk.MockKAnnotations
import io.mockk.every
import io.mockk.impl.annotations.MockK
import io.mockk.mockk
import io.mockk.mockkConstructor
import io.mockk.mockkStatic
import io.mockk.unmockkAll
import io.mockk.unmockkConstructor
import io.mockk.verify
import kotlinx.coroutines.runBlocking
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class AuthManagerTest {
    @MockK
    lateinit var auth: FirebaseAuth

    @MockK
    lateinit var firebaseUser: FirebaseUser

    private lateinit var authManager: AuthManager
    private lateinit var context: Context

    @Before
    fun setUp() {
        MockKAnnotations.init(this)
        context = ApplicationProvider.getApplicationContext()

        if (FirebaseApp.getApps(context).isEmpty()) {
            val options =
                FirebaseOptions
                    .Builder()
                    .setApplicationId("com.itlab.openvino")
                    .setApiKey("fake_api_key")
                    .setProjectId("fake_project_id")
                    .build()
            FirebaseApp.initializeApp(context, options)
        }

        authManager = AuthManager(auth)
        mockkStatic(AuthUI::class)
    }

    @After
    fun tearDown() {
        unmockkAll()
    }

    @Test
    fun `getCurrentUserId should return uid when user is logged in`() {
        every { auth.currentUser } returns firebaseUser
        every { firebaseUser.uid } returns "test_user_123"

        val result = authManager.getCurrentUserId()

        assertEquals("test_user_123", result)
    }

    @Test
    fun `getCurrentUserId should return null when no user`() {
        every { auth.currentUser } returns null
        assertNull(authManager.getCurrentUserId())
    }

    @Test
    fun `signOut should call AuthUI signOut`() =
        runBlocking {
            val authUI = mockk<AuthUI>()

            val mockTask = Tasks.forResult<Void>(null)

            every { AuthUI.getInstance() } returns authUI
            every { authUI.signOut(any()) } returns mockTask

            authManager.signOut(context)

            verify { authUI.signOut(context) }
        }

    @Test
    fun `getSignInIntent should return intent from builder`() {
        mockkStatic(AuthUI::class)
        val mockAuthUI = mockk<AuthUI>()
        val mockBuilder = mockk<AuthUI.SignInIntentBuilder>(relaxed = true)
        val expectedIntent = Intent()

        every { AuthUI.getInstance() } returns mockAuthUI
        every { mockAuthUI.createSignInIntentBuilder() } returns mockBuilder

        mockkConstructor(AuthUI.IdpConfig.EmailBuilder::class)
        mockkConstructor(AuthUI.IdpConfig.GoogleBuilder::class)

        val mockEmailConfig = mockk<AuthUI.IdpConfig>()
        val mockGoogleConfig = mockk<AuthUI.IdpConfig>()

        every { anyConstructed<AuthUI.IdpConfig.EmailBuilder>().build() } returns mockEmailConfig
        every { anyConstructed<AuthUI.IdpConfig.GoogleBuilder>().build() } returns mockGoogleConfig
        every { mockBuilder.setAvailableProviders(any()) } returns mockBuilder
        every { mockBuilder.setIsSmartLockEnabled(any()) } returns mockBuilder
        every { mockBuilder.build() } returns expectedIntent

        val result = authManager.getSignInIntent()

        assertEquals(expectedIntent, result)
        verify { mockBuilder.build() }

        unmockkConstructor(AuthUI.IdpConfig.EmailBuilder::class)
        unmockkConstructor(AuthUI.IdpConfig.GoogleBuilder::class)
    }
}

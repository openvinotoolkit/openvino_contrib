package com.itlab.domain

import com.itlab.domain.cloud.SyncState
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class SyncStateTest {
    @Test
    fun `verify all sync states`() {
        val idle = SyncState.Idle
        assertTrue(idle is SyncState)

        val syncing = SyncState.Syncing
        assertTrue(syncing is SyncState)

        val success = SyncState.Success
        assertTrue(success is SyncState)

        val errorMessage = "Network timeout"
        val error = SyncState.Error(errorMessage)
        assertEquals(errorMessage, error.message)
    }
}

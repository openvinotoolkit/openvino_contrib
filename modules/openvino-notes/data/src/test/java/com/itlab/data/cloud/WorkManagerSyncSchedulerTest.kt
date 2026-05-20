package com.itlab.data.cloud

import androidx.work.ExistingWorkPolicy
import androidx.work.OneTimeWorkRequest
import androidx.work.WorkManager
import io.mockk.every
import io.mockk.mockk
import io.mockk.slot
import io.mockk.verify
import org.junit.Assert.assertNotNull
import org.junit.Test

class WorkManagerSyncSchedulerTest {
    private val workManager: WorkManager = mockk(relaxed = true)
    private val scheduler = WorkManagerSyncScheduler(workManager)

    @Test
    fun `scheduleSync should enqueue unique work with correct parameters`() {
        val userId = "test_user_777"
        val expectedWorkName = "sync_work_$userId"

        val workRequestSlot = slot<OneTimeWorkRequest>()

        every {
            workManager.enqueueUniqueWork(
                eq(expectedWorkName),
                eq(ExistingWorkPolicy.KEEP),
                capture(workRequestSlot),
            )
        } returns mockk(relaxed = true)

        scheduler.scheduleSync(userId)

        verify(exactly = 1) {
            workManager.enqueueUniqueWork(
                expectedWorkName,
                ExistingWorkPolicy.KEEP,
                any<OneTimeWorkRequest>(),
            )
        }

        val capturedRequest = workRequestSlot.captured
        assertNotNull(capturedRequest)
    }
}

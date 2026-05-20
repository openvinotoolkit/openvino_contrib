package com.itlab.domain.cloud

interface SyncScheduler {
    fun scheduleSync(userId: String)
}

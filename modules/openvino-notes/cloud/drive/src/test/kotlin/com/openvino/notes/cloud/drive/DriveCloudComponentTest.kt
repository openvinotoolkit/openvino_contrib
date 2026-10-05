// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package com.openvino.notes.cloud.drive

import com.openvino.notes.cloud.api.RemoteErrorCode
import com.openvino.notes.cloud.api.RemoteObjectId
import com.openvino.notes.cloud.api.RemoteOutcome
import com.openvino.notes.cloud.api.UploadDescriptor
import com.openvino.notes.identity.api.FakeIdentityService
import com.openvino.notes.kernel.AccountKey
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Test

class DriveCloudComponentTest {
    @Test fun `initial discovery and resumable transfer expose typed unavailable outcomes`() = runTest {
        val component = DriveCloudComponent.create(FakeIdentityService())
        val accountKey = AccountKey("account")

        assertFailureCode(RemoteErrorCode.SIGNED_OUT, component.objectStore.list(accountKey))
        assertFailureCode(RemoteErrorCode.SIGNED_OUT, component.changeFeed.startCursor(accountKey))
        assertFailureCode(
            RemoteErrorCode.SIGNED_OUT,
            component.transferClient.startUpload(
                accountKey,
                UploadDescriptor(RemoteObjectId("media"), "image.jpg", "image/jpeg", 1024),
            ),
        )
    }

    private fun assertFailureCode(expected: RemoteErrorCode, outcome: RemoteOutcome<*>) {
        assertEquals(expected, (outcome as RemoteOutcome.Failure).error.code)
    }
}

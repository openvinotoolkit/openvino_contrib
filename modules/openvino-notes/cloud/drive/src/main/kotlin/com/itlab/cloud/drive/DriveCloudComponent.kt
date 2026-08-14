// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package com.itlab.cloud.drive

import com.itlab.cloud.api.RemoteChangeFeed
import com.itlab.cloud.api.RemoteChangePage
import com.itlab.cloud.api.RemoteCursor
import com.itlab.cloud.api.RemoteError
import com.itlab.cloud.api.RemoteErrorCode
import com.itlab.cloud.api.RemoteObject
import com.itlab.cloud.api.RemoteObjectId
import com.itlab.cloud.api.RemoteObjectMetadata
import com.itlab.cloud.api.RemoteObjectStore
import com.itlab.cloud.api.RemoteOutcome
import com.itlab.identity.api.AccessTokenOutcome
import com.itlab.identity.api.AccessTokenProvider
import com.itlab.kernel.AccountKey

internal class DriveRemoteStore(private val tokens: AccessTokenProvider) : RemoteObjectStore, RemoteChangeFeed {
    override suspend fun put(
        accountKey: AccountKey,
        objectValue: RemoteObject,
        expectedRevision: com.itlab.cloud.api.RemoteRevision?,
    ): RemoteOutcome<RemoteObjectMetadata> = unavailable()
    override suspend fun get(accountKey: AccountKey, id: RemoteObjectId): RemoteOutcome<RemoteObject> = unavailable()
    override suspend fun delete(
        accountKey: AccountKey,
        id: RemoteObjectId,
        expectedRevision: com.itlab.cloud.api.RemoteRevision?,
    ): RemoteOutcome<RemoteObjectMetadata> = unavailable()
    override suspend fun changes(accountKey: AccountKey, cursor: RemoteCursor?): RemoteOutcome<RemoteChangePage> = unavailable()

    private suspend fun unavailable(): RemoteOutcome.Failure = when (tokens.accessToken()) {
        AccessTokenOutcome.SignedOut -> failure(RemoteErrorCode.SIGNED_OUT, "drive.signed_out")
        AccessTokenOutcome.NotAuthorized -> failure(RemoteErrorCode.NOT_AUTHORIZED, "drive.not_authorized")
        is AccessTokenOutcome.Failed -> failure(RemoteErrorCode.NOT_CONFIGURED, "drive.client_not_configured")
        is AccessTokenOutcome.Available -> failure(RemoteErrorCode.NOT_CONFIGURED, "drive.transport_not_connected")
    }

    private fun failure(code: RemoteErrorCode, diagnosticCode: String) = RemoteOutcome.Failure(RemoteError(code, diagnosticCode))
}

data class DriveCloudComponent(val objectStore: RemoteObjectStore, val changeFeed: RemoteChangeFeed) {
    companion object {
        fun create(tokens: AccessTokenProvider): DriveCloudComponent {
            val store = DriveRemoteStore(tokens)
            return DriveCloudComponent(store, store)
        }
    }
}

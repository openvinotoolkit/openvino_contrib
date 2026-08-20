// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package com.openvino.notes.cloud.drive

import com.openvino.notes.cloud.api.RemoteChangeFeed
import com.openvino.notes.cloud.api.RemoteChangePage
import com.openvino.notes.cloud.api.RemoteCursor
import com.openvino.notes.cloud.api.RemoteError
import com.openvino.notes.cloud.api.RemoteErrorCode
import com.openvino.notes.cloud.api.RemoteObject
import com.openvino.notes.cloud.api.RemoteObjectId
import com.openvino.notes.cloud.api.RemoteObjectMetadata
import com.openvino.notes.cloud.api.RemoteObjectStore
import com.openvino.notes.cloud.api.RemoteOutcome
import com.openvino.notes.cloud.api.RemoteByteSink
import com.openvino.notes.cloud.api.ResumableTransferClient
import com.openvino.notes.cloud.api.TransferSessionId
import com.openvino.notes.cloud.api.UploadChunk
import com.openvino.notes.cloud.api.UploadChunkOutcome
import com.openvino.notes.cloud.api.UploadDescriptor
import com.openvino.notes.cloud.api.UploadSession
import com.openvino.notes.identity.api.port.AccessTokenOutcome
import com.openvino.notes.identity.api.port.AccessTokenProvider
import com.openvino.notes.kernel.AccountKey

internal class DriveRemoteStore(private val tokens: AccessTokenProvider) :
    RemoteObjectStore,
    RemoteChangeFeed,
    ResumableTransferClient {
    override suspend fun list(accountKey: AccountKey): RemoteOutcome<List<RemoteObjectMetadata>> = unavailable()
    override suspend fun put(
        accountKey: AccountKey,
        objectValue: RemoteObject,
        expectedRevision: com.openvino.notes.cloud.api.RemoteRevision?,
    ): RemoteOutcome<RemoteObjectMetadata> = unavailable()
    override suspend fun get(accountKey: AccountKey, id: RemoteObjectId): RemoteOutcome<RemoteObject> = unavailable()
    override suspend fun delete(
        accountKey: AccountKey,
        id: RemoteObjectId,
        expectedRevision: com.openvino.notes.cloud.api.RemoteRevision?,
    ): RemoteOutcome<RemoteObjectMetadata> = unavailable()
    override suspend fun startCursor(accountKey: AccountKey): RemoteOutcome<RemoteCursor> = unavailable()
    override suspend fun changes(accountKey: AccountKey, cursor: RemoteCursor): RemoteOutcome<RemoteChangePage> = unavailable()
    override suspend fun startUpload(
        accountKey: AccountKey,
        descriptor: UploadDescriptor,
        expectedRevision: com.openvino.notes.cloud.api.RemoteRevision?,
    ): RemoteOutcome<UploadSession> = unavailable()
    override suspend fun resumeUpload(
        accountKey: AccountKey,
        sessionId: TransferSessionId,
    ): RemoteOutcome<UploadSession> = unavailable()
    override suspend fun uploadChunk(
        accountKey: AccountKey,
        sessionId: TransferSessionId,
        chunk: UploadChunk,
    ): RemoteOutcome<UploadChunkOutcome> = unavailable()
    override suspend fun downloadStream(
        accountKey: AccountKey,
        id: RemoteObjectId,
        sink: RemoteByteSink,
    ): RemoteOutcome<RemoteObjectMetadata> = unavailable()

    private suspend fun unavailable(): RemoteOutcome.Failure = when (tokens.accessToken()) {
        AccessTokenOutcome.SignedOut -> failure(RemoteErrorCode.SIGNED_OUT, "drive.signed_out")
        AccessTokenOutcome.NotAuthorized -> failure(RemoteErrorCode.NOT_AUTHORIZED, "drive.not_authorized")
        is AccessTokenOutcome.Failed -> failure(RemoteErrorCode.NOT_CONFIGURED, "drive.client_not_configured")
        is AccessTokenOutcome.Available -> failure(RemoteErrorCode.NOT_CONFIGURED, "drive.transport_not_connected")
    }

    private fun failure(code: RemoteErrorCode, diagnosticCode: String) = RemoteOutcome.Failure(RemoteError(code, diagnosticCode))
}

data class DriveCloudComponent(
    val objectStore: RemoteObjectStore,
    val changeFeed: RemoteChangeFeed,
    val transferClient: ResumableTransferClient,
) {
    companion object {
        fun create(tokens: AccessTokenProvider): DriveCloudComponent {
            val store = DriveRemoteStore(tokens)
            return DriveCloudComponent(store, store, store)
        }
    }
}

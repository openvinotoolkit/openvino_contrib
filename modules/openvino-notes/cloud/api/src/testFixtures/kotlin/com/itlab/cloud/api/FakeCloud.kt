// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package com.itlab.cloud.api

import com.itlab.identity.api.AccountId

class FakeRemoteStore : RemoteObjectStore, RemoteChangeFeed {
    val objects = linkedMapOf<RemoteObjectId, RemoteObject>()
    var changeOutcome: RemoteOutcome<RemoteChangePage> = RemoteOutcome.Success(RemoteChangePage(emptyList(), null))
    override suspend fun put(accountId: AccountId, objectValue: RemoteObject): RemoteOutcome<RemoteObjectMetadata> {
        objects[objectValue.id] = objectValue
        return RemoteOutcome.Success(RemoteObjectMetadata(objectValue.id, objectValue.name, objectValue.mediaType, objectValue.modifiedAt))
    }
    override suspend fun get(accountId: AccountId, id: RemoteObjectId): RemoteOutcome<RemoteObject> =
        objects[id]?.let { RemoteOutcome.Success(it) }
            ?: RemoteOutcome.Failure(RemoteError(RemoteErrorCode.NOT_FOUND, "fake.not_found"))
    override suspend fun delete(accountId: AccountId, id: RemoteObjectId): RemoteOutcome<Unit> {
        objects.remove(id)
        return RemoteOutcome.Success(Unit)
    }
    override suspend fun changes(accountId: AccountId, cursor: RemoteCursor?): RemoteOutcome<RemoteChangePage> = changeOutcome
}

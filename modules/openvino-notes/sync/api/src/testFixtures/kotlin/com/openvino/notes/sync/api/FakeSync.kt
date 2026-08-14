// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package com.openvino.notes.sync.api

import kotlinx.coroutines.flow.MutableStateFlow

class FakeSyncService(initial: SyncState = SyncState()) : SyncService {
    override val state = MutableStateFlow(initial)
    var outcome: SyncOutcome = SyncOutcome.Completed(0, 0)
    override suspend fun sync(reason: SyncReason): SyncOutcome = outcome
}

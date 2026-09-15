// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package com.openvino.notes.notes.core

import com.openvino.notes.kernel.AccountKey
import com.openvino.notes.kernel.AccountScope
import kotlinx.coroutines.flow.MutableStateFlow

internal class FakeAccountScope(initial: AccountKey? = null) : AccountScope {
    override val activeAccountKey = MutableStateFlow(initial)
    override fun currentAccountKey(): AccountKey? = activeAccountKey.value
    fun set(accountKey: AccountKey?) { activeAccountKey.value = accountKey }
}

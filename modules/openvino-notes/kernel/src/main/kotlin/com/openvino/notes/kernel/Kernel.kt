// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package com.openvino.notes.kernel

import java.time.Instant
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow

@JvmInline
value class AccountKey(val value: String) {
    init {
        require(value.isNotBlank()) { "AccountKey must not be blank" }
    }
}

/** Neutral active-account boundary shared by capabilities without depending on Identity. */
interface AccountScope {
    val activeAccountKey: Flow<AccountKey?>
    fun currentAccountKey(): AccountKey?
}

fun interface AppClock {
    fun now(): Instant
}

object SystemAppClock : AppClock {
    override fun now(): Instant = Instant.now()
}

data class AppDispatchers(
    val io: CoroutineDispatcher,
    val default: CoroutineDispatcher,
) {
    companion object {
        fun production(): AppDispatchers = AppDispatchers(Dispatchers.IO, Dispatchers.Default)
    }
}

@JvmInline
value class OperationId(val value: String)

enum class DiagnosticLevel { DEBUG, INFO, WARNING, ERROR }

data class DiagnosticEvent(
    val code: String,
    val level: DiagnosticLevel,
    val attributes: Map<String, String> = emptyMap(),
)

fun interface AppLogger {
    fun log(event: DiagnosticEvent)
}

object NoOpAppLogger : AppLogger {
    override fun log(event: DiagnosticEvent) = Unit
}

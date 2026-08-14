// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package com.itlab.sync.core

import com.itlab.cloud.api.RemoteChangeFeed
import com.itlab.cloud.api.RemoteCursor
import com.itlab.cloud.api.RemoteError
import com.itlab.cloud.api.RemoteErrorCode
import com.itlab.cloud.api.RemoteObject
import com.itlab.cloud.api.RemoteObjectId
import com.itlab.cloud.api.RemoteObjectStore
import com.itlab.cloud.api.RemoteOutcome
import com.itlab.identity.api.SessionReader
import com.itlab.kernel.AppClock
import com.itlab.kernel.AppLogger
import com.itlab.kernel.DiagnosticEvent
import com.itlab.kernel.DiagnosticLevel
import com.itlab.notes.api.LocalChangeKind
import com.itlab.notes.api.LocalNoteChange
import com.itlab.notes.api.FolderId
import com.itlab.notes.api.Note
import com.itlab.notes.api.NoteId
import com.itlab.notes.api.NotesSyncPort
import com.itlab.notes.api.RemoteNoteChange
import com.itlab.sync.api.SyncExecutor
import com.itlab.sync.api.SyncOutcome
import com.itlab.sync.api.SyncPhase
import com.itlab.sync.api.SyncReason
import com.itlab.sync.api.SyncService
import com.itlab.sync.api.SyncState
import com.itlab.sync.api.SyncStateStore
import java.nio.charset.StandardCharsets
import java.time.Instant
import java.util.Base64
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow

class DefaultSyncService(
    private val sessions: SessionReader,
    private val notes: NotesSyncPort,
    private val objects: RemoteObjectStore,
    private val changes: RemoteChangeFeed,
    private val stateStore: SyncStateStore,
    private val clock: AppClock,
    private val logger: AppLogger,
) : SyncService, SyncExecutor {
    private val mutableState = MutableStateFlow(SyncState())
    override val state: StateFlow<SyncState> = mutableState
    override suspend fun execute(reason: SyncReason): SyncOutcome = sync(reason)

    override suspend fun sync(reason: SyncReason): SyncOutcome {
        val accountId = sessions.currentAccountId() ?: return finish(SyncOutcome.SignedOut, "sync.signed_out")
        val persisted = stateStore.read(accountId)
        mutableState.value = persisted.copy(phase = SyncPhase.RUNNING, diagnosticCode = null)
        logger.log(DiagnosticEvent("sync.started", DiagnosticLevel.INFO, mapOf("reason" to reason.name)))

        val pending = notes.pendingChanges(accountId)
        val acknowledged = linkedSetOf<String>()
        for (change in pending) {
            val result = when (change.kind) {
                LocalChangeKind.UPSERT -> objects.put(accountId, change.toRemoteObject())
                LocalChangeKind.DELETE -> objects.delete(accountId, RemoteObjectId(change.noteId.value))
            }
            when (result) {
                is RemoteOutcome.Success -> acknowledged += change.changeId
                is RemoteOutcome.Failure -> return finish(result.error.toSyncOutcome(), result.error.diagnosticCode, accountId)
            }
        }
        notes.acknowledge(acknowledged)

        val page = when (val result = changes.changes(accountId, persisted.continuationToken?.let(::RemoteCursor))) {
            is RemoteOutcome.Success -> result.value
            is RemoteOutcome.Failure -> return finish(result.error.toSyncOutcome(), result.error.diagnosticCode, accountId)
        }
        val remoteChanges = mutableListOf<RemoteNoteChange>()
        for (metadata in page.changes) {
            if (metadata.deleted) {
                remoteChanges += RemoteNoteChange(metadata.id.value, NoteId(metadata.name), true, metadata.modifiedAt)
                continue
            }
            when (val result = objects.get(accountId, metadata.id)) {
                is RemoteOutcome.Success -> remoteChanges += result.value.toRemoteChange(accountId.value)
                is RemoteOutcome.Failure -> return finish(result.error.toSyncOutcome(), result.error.diagnosticCode, accountId)
            }
        }
        notes.applyRemote(accountId, remoteChanges)

        val completedState = SyncState(
            phase = SyncPhase.IDLE,
            lastSuccessfulAt = clock.now(),
            continuationToken = page.nextCursor?.value,
        )
        mutableState.value = completedState
        stateStore.write(accountId, completedState)
        logger.log(DiagnosticEvent("sync.completed", DiagnosticLevel.INFO, mapOf("uploaded" to acknowledged.size.toString(), "downloaded" to remoteChanges.size.toString())))
        return SyncOutcome.Completed(acknowledged.size, remoteChanges.size)
    }

    private suspend fun finish(outcome: SyncOutcome, code: String, accountId: com.itlab.identity.api.AccountId? = null): SyncOutcome {
        val phase = if (outcome is SyncOutcome.Failed) SyncPhase.FAILED else SyncPhase.BLOCKED
        mutableState.value = mutableState.value.copy(phase = phase, diagnosticCode = code)
        if (accountId != null) stateStore.write(accountId, mutableState.value)
        logger.log(DiagnosticEvent(code, DiagnosticLevel.WARNING))
        return outcome
    }
}

data class SyncCoreComponent(val service: SyncService, val executor: SyncExecutor) {
    companion object {
        fun create(
            sessions: SessionReader,
            notes: NotesSyncPort,
            objects: RemoteObjectStore,
            changes: RemoteChangeFeed,
            stateStore: SyncStateStore,
            clock: AppClock,
            logger: AppLogger,
        ): SyncCoreComponent {
            val service = DefaultSyncService(sessions, notes, objects, changes, stateStore, clock, logger)
            return SyncCoreComponent(service, service)
        }
    }
}

private fun RemoteError.toSyncOutcome(): SyncOutcome = when (code) {
    RemoteErrorCode.SIGNED_OUT -> SyncOutcome.SignedOut
    RemoteErrorCode.NOT_CONFIGURED, RemoteErrorCode.NOT_AUTHORIZED -> SyncOutcome.Blocked(diagnosticCode)
    RemoteErrorCode.UNAVAILABLE -> SyncOutcome.Failed(diagnosticCode, retryable = true)
    else -> SyncOutcome.Failed(diagnosticCode, retryable = false)
}

private fun LocalNoteChange.toRemoteObject(): RemoteObject {
    val note = requireNotNull(payload) { "UPSERT change must carry a note snapshot" }
    return RemoteObject(RemoteObjectId(note.id.value), note.id.value, NOTE_MEDIA_TYPE, note.updatedAt, NoteCodec.encode(note))
}

private fun RemoteObject.toRemoteChange(expectedAccountId: String): RemoteNoteChange {
    val note = NoteCodec.decode(bytes)
    require(note.accountId.value == expectedAccountId) { "Remote note belongs to a different account" }
    return RemoteNoteChange(id.value, note.id, false, modifiedAt, note)
}

private const val NOTE_MEDIA_TYPE = "application/vnd.openvino-notes.note-v1"

private object NoteCodec {
    fun encode(note: Note): ByteArray {
        val encoder = Base64.getUrlEncoder().withoutPadding()
        fun field(value: String): String = encoder.encodeToString(value.toByteArray(StandardCharsets.UTF_8))
        return listOf(note.id.value, note.accountId.value, note.updatedAt.toEpochMilli().toString(), field(note.title), field(note.body), field(note.folderId?.value.orEmpty())).joinToString("\n").toByteArray(StandardCharsets.UTF_8)
    }

    fun decode(bytes: ByteArray): Note {
        val fields = bytes.toString(StandardCharsets.UTF_8).split('\n')
        require(fields.size == 6) { "Invalid note payload" }
        val decoder = Base64.getUrlDecoder()
        fun field(value: String): String = String(decoder.decode(value), StandardCharsets.UTF_8)
        return Note(
            id = NoteId(fields[0]),
            accountId = com.itlab.identity.api.AccountId(fields[1]),
            title = field(fields[3]),
            body = field(fields[4]),
            folderId = field(fields[5]).takeIf(String::isNotEmpty)?.let(::FolderId),
            updatedAt = Instant.ofEpochMilli(fields[2].toLong()),
        )
    }
}

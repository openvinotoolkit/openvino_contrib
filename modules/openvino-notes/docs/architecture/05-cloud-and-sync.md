<!--
Copyright (C) 2018-2026 Intel Corporation
SPDX-License-Identifier: Apache-2.0
-->

# Cloud and Synchronization

## Cloud Contracts

`:cloud:api` describes provider-neutral remote storage. `RemoteObjectStore`
supports initial listing plus revision-aware put/get/delete operations.
`RemoteChangeFeed.startCursor()` creates the first durable change cursor;
`changes(cursor)` requires a non-null cursor. This makes initial discovery and
incremental polling separate operations and avoids ambiguous null-cursor logic.

Small metadata objects may use the generic `RemoteObject` value. Media uses
`ResumableTransferClient`, which separates:

- starting or resuming a session;
- uploading a bounded chunk at a declared offset;
- observing `InProgress(nextOffset)`;
- observing `Completed(metadata)` with durable object metadata and revision;
- streaming download chunks without materializing the full object.

The Drive module translates these contracts to a provider implementation. It is
currently an adapter placeholder and returns typed unavailable outcomes.

## Public State Versus Internal State

`SyncState` is a small consumer model suitable for UI: idle, running progress,
blocked reason, or failure. It intentionally contains no remote cursor or session
capability. Internal durable state is split into two ports:

- `SyncCheckpointPort` stores the change cursor, known revisions, last successful
  time, and reset requirements for an account.
- `SyncTransferCheckpointPort` stores operation/object identity, redacted upload
  session capability, next offset, and expiry for resumable transfers.

`:sync:android` implements both with Room. The split lets `:sync:core` remain a
plain JVM module and prevents UI consumers from depending on replication details.

## Required Remote-First Algorithm

The production engine is not implemented, but its safe order is constrained by
the contracts:

1. Load the explicit account's replication and transfer checkpoints.
2. Establish an initial listing/cursor or fetch changes after the stored cursor.
3. Validate and apply remote changes through Notes repository ports. Remote apply
   must not create local outbox operations.
4. Resume any persisted transfer from its recorded offset, or start a new
   session and persist it before sending chunks.
5. Upload local operations. Treat completion metadata/revision as the remote
   acknowledgement; do not infer success from bytes written alone.
6. Persist the remote revision and relevant checkpoint transactionally.
7. Remove the transfer checkpoint, then acknowledge the local outbox operation.
8. Advance the change cursor only after all preceding state is durable.

On conflicts, cursor reset, token rejection, expiry, malformed remote data, or
cancellation, preserve enough durable state to retry safely. Never acknowledge a
local operation before remote completion and local revision persistence.

## WorkManager Isolation

The scheduler creates a unique work name and tags derived from a non-secret hash
of `AccountKey`. The immutable account key is also placed in Worker input. The
Worker restores that input and invokes the executor for the same account; it does
not consult whichever account happens to be active in the UI. Missing or invalid
account input fails closed. Account-specific cancellation must not cancel another
account's work.

## Current Runtime Behavior

`DisabledSyncService` returns `Blocked(NotConfigured)` before reading or mutating
local or remote data. This is intentional: an upload-first partial implementation
could lose conflict information or falsely acknowledge outbox entries. Enabling
sync requires the complete remote-first algorithm, provider integration, database
reopen tests for checkpoints, conflict/reset coverage, and cancellation tests at
chunk and transaction boundaries.

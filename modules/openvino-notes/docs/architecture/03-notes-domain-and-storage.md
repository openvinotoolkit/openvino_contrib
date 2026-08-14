<!--
Copyright (C) 2018-2026 Intel Corporation
SPDX-License-Identifier: Apache-2.0
-->

# Notes Domain and Storage

## Aggregate Model

`Note` is the durable aggregate presented to consumers. It contains a validated
`NoteId`, title, ordered `ContentItem` list, attachment metadata, optional folder,
tags, favorite flag, optional summary, and creation/update timestamps. Content is
structured rather than flattened:

- `Text` holds editable text;
- `Image` references attachment metadata;
- `File` references an attachment without treating it as an image;
- `Link` retains a link as a distinct item.

`Folder` is a first-class Notes capability with create, rename, delete, and
move-note operations. IDs (`NoteId`, `ContentItemId`, `AttachmentId`, and
`FolderId`) are typed so unrelated identifiers cannot be interchanged.

Updates use patch semantics through `FieldUpdate.Keep` and `FieldUpdate.Set`.
Omitted fields therefore remain unchanged; a caller cannot accidentally erase
structured content while changing the title or summary. The editor updates the
targeted text item and preserves images, files, links, and other text items.

## Domain Invariants

`:notes:core` owns policy and rejects invalid writes before persistence:

- the title must be non-blank and no longer than 200 characters;
- aggregate text is bounded to 100,000 characters;
- content and attachment identifiers are unique within a Note;
- every media content item references attachment metadata owned by that Note;
- an assigned folder must exist in the same account scope;
- folder names satisfy the capability's naming rules;
- deleting a non-empty folder is rejected until notes are moved.

Each operation receives an explicit `AccountKey`. Repositories are ports under
`:notes:api`; policy does not import Room or Identity. Typed outcomes distinguish
validation, not-found, conflict, storage, and unavailable failures rather than
encoding failure as an empty model.

## Transactional Local Persistence

`:notes:room` maps domain objects to a versioned Room schema. Local note and
folder writes atomically update the aggregate tables and append an outbox
operation. Replication metadata includes remote revisions, tombstones, malformed
remote records, and conflict state. Applying a remote change updates local state
without emitting a new local outbox operation, which prevents replication loops.

Every query and row is account-partitioned. A repository operation must use the
requested account in all keys and predicates; observing a new active session must
not expose the previous account's Notes data.

Room schema JSON is committed below each Room adapter. Once a schema version has
shipped, incrementing it requires an explicit migration plus tests from the last
committed version. Destructive fallback is not an accepted migration strategy.

## Attachment Ownership

Room owns attachment metadata and account-scoped files below the application's
`files/notes-media/` area. `AttachmentContentPort` is the only binary access seam;
it exposes no absolute path. `BinarySource` offers bounded offset reads and a
stable declared size so consumers can stream data and honor cancellation.

An `AttachmentId` identifies immutable bytes:

1. The first write streams 64 KiB chunks into a temporary file.
2. The adapter verifies size/content and atomically publishes the completed file.
3. Repeating the same bytes for the same ID is idempotent.
4. Different bytes for an existing ID are rejected.
5. Replacing content requires a new `AttachmentId` and a Note update.

Deletion and remote replacement remove files no longer referenced by the owning
Note. Callers must not cache file paths or bypass the port, because cleanup and
account isolation belong to the storage adapter.

## Replication Boundary

The Notes API also owns repository-facing remote-change and local-outbox models,
because they describe how Notes state may be read or applied. Sync owns the
orchestration and checkpoints, not the internal Note schema. This separation lets
Room be replaced without changing the eventual sync algorithm and lets sync tests
use in-memory Notes fakes.

// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package com.itlab.assistant.api

import com.itlab.notes.api.AttachmentId
import com.itlab.notes.api.ContentItemId
import com.itlab.notes.api.NoteId

class FakeNoteAssistant : NoteAssistant {
    var outcome: SuggestionOutcome = SuggestionOutcome.Unavailable("fake not configured")
    override suspend fun summarize(noteId: NoteId): SuggestionOutcome = outcome
    override suspend fun suggestTextTags(noteId: NoteId): SuggestionOutcome = outcome
    override suspend fun rewrite(
        noteId: NoteId,
        contentItemId: ContentItemId,
        style: AssistantRewriteStyle,
    ): SuggestionOutcome = outcome
    override suspend fun suggestImageTags(
        noteId: NoteId,
        attachmentId: AttachmentId,
        bytes: ByteArray,
        mediaType: String,
    ): SuggestionOutcome = outcome
    override suspend fun apply(suggestion: AssistantSuggestion): ApplySuggestionOutcome = ApplySuggestionOutcome.Applied
}

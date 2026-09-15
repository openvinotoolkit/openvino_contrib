// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package com.openvino.notes.assistant.api

import com.openvino.notes.notes.api.AttachmentId
import com.openvino.notes.notes.api.ContentItemId
import com.openvino.notes.notes.api.NoteId

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
    ): SuggestionOutcome = outcome
    override suspend fun apply(suggestion: AssistantSuggestion): ApplySuggestionOutcome = ApplySuggestionOutcome.Applied
}

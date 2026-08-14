// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package com.itlab.assistant.api

import com.itlab.notes.api.AttachmentId
import com.itlab.notes.api.ContentItemId
import com.itlab.notes.api.NoteId
import com.itlab.notes.api.NoteTag

enum class AssistantRewriteStyle { CONCISE, CLEAR, PROFESSIONAL, CASUAL }

sealed interface AssistantSuggestion {
    val noteId: NoteId

    data class Summary(override val noteId: NoteId, val value: String) : AssistantSuggestion
    data class TextTags(override val noteId: NoteId, val values: Set<NoteTag>) : AssistantSuggestion
    data class Rewrite(
        override val noteId: NoteId,
        val contentItemId: ContentItemId,
        val value: String,
    ) : AssistantSuggestion
    data class ImageTags(
        override val noteId: NoteId,
        val attachmentId: AttachmentId,
        val values: Set<NoteTag>,
    ) : AssistantSuggestion
}

sealed interface SuggestionOutcome {
    data class Ready(val suggestion: AssistantSuggestion) : SuggestionOutcome
    data object NoteNotFound : SuggestionOutcome
    data class InvalidTarget(val reason: String) : SuggestionOutcome
    data class Unavailable(val reason: String) : SuggestionOutcome
    data class Failed(val code: String) : SuggestionOutcome
}

sealed interface ApplySuggestionOutcome {
    data object Applied : ApplySuggestionOutcome
    data object NoteNotFound : ApplySuggestionOutcome
    data class Rejected(val reason: String) : ApplySuggestionOutcome
}

interface NoteAssistant {
    suspend fun summarize(noteId: NoteId): SuggestionOutcome
    suspend fun suggestTextTags(noteId: NoteId): SuggestionOutcome
    suspend fun rewrite(
        noteId: NoteId,
        contentItemId: ContentItemId,
        style: AssistantRewriteStyle,
    ): SuggestionOutcome
    suspend fun suggestImageTags(
        noteId: NoteId,
        attachmentId: AttachmentId,
        bytes: ByteArray,
        mediaType: String,
    ): SuggestionOutcome
    suspend fun apply(suggestion: AssistantSuggestion): ApplySuggestionOutcome
}

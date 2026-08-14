// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package com.openvino.notes.assistant.core

import com.openvino.notes.ai.image.api.ImageInput
import com.openvino.notes.ai.image.api.ImageOutcome
import com.openvino.notes.ai.image.api.ImageTagger
import com.openvino.notes.ai.text.api.RewriteOutcome
import com.openvino.notes.ai.text.api.RewriteRequest
import com.openvino.notes.ai.text.api.RewriteStyle
import com.openvino.notes.ai.text.api.SummaryOutcome
import com.openvino.notes.ai.text.api.SummaryRequest
import com.openvino.notes.ai.text.api.TextAssistant
import com.openvino.notes.ai.text.api.TextTagsOutcome
import com.openvino.notes.ai.text.api.TextTagsRequest
import com.openvino.notes.assistant.api.ApplySuggestionOutcome
import com.openvino.notes.assistant.api.AssistantRewriteStyle
import com.openvino.notes.assistant.api.AssistantSuggestion
import com.openvino.notes.assistant.api.NoteAssistant
import com.openvino.notes.assistant.api.SuggestionOutcome
import com.openvino.notes.notes.api.AttachmentContentPort
import com.openvino.notes.notes.api.AttachmentId
import com.openvino.notes.notes.api.ContentItem
import com.openvino.notes.notes.api.ContentItemId
import com.openvino.notes.notes.api.FieldUpdate
import com.openvino.notes.notes.api.Note
import com.openvino.notes.notes.api.NoteId
import com.openvino.notes.notes.api.NoteMutationOutcome
import com.openvino.notes.notes.api.NoteTag
import com.openvino.notes.notes.api.NotesService
import com.openvino.notes.notes.api.UpdateNoteCommand
import kotlinx.coroutines.CancellationException

class DefaultNoteAssistant(
    private val notes: NotesService,
    private val attachmentContent: AttachmentContentPort,
    private val text: TextAssistant,
    private val images: ImageTagger,
) : NoteAssistant {
    override suspend fun summarize(noteId: NoteId): SuggestionOutcome = withNote(noteId) { note ->
        when (val outcome = text.summarize(SummaryRequest(note.plainText()))) {
            is SummaryOutcome.Generated -> ready(AssistantSuggestion.Summary(noteId, outcome.summary.trim()))
            is SummaryOutcome.NotConfigured -> SuggestionOutcome.Unavailable(outcome.reason)
            is SummaryOutcome.Failed -> SuggestionOutcome.Failed(outcome.code)
        }
    }

    override suspend fun suggestTextTags(noteId: NoteId): SuggestionOutcome = withNote(noteId) { note ->
        when (val outcome = text.suggestTags(TextTagsRequest(note.plainText()))) {
            is TextTagsOutcome.Generated -> ready(
                AssistantSuggestion.TextTags(noteId, outcome.tags.map(::NoteTag).toSet()),
            )
            is TextTagsOutcome.NotConfigured -> SuggestionOutcome.Unavailable(outcome.reason)
            is TextTagsOutcome.Failed -> SuggestionOutcome.Failed(outcome.code)
        }
    }

    override suspend fun rewrite(
        noteId: NoteId,
        contentItemId: ContentItemId,
        style: AssistantRewriteStyle,
    ): SuggestionOutcome = withNote(noteId) { note ->
        val item = note.contentItems.firstOrNull { it.id == contentItemId } as? ContentItem.Text
            ?: return@withNote SuggestionOutcome.InvalidTarget("content item is not text")
        when (val outcome = text.rewrite(RewriteRequest(item.text, style.toTextStyle()))) {
            is RewriteOutcome.Generated -> ready(
                AssistantSuggestion.Rewrite(noteId, contentItemId, outcome.content.trim()),
            )
            is RewriteOutcome.NotConfigured -> SuggestionOutcome.Unavailable(outcome.reason)
            is RewriteOutcome.Failed -> SuggestionOutcome.Failed(outcome.code)
        }
    }

    override suspend fun suggestImageTags(
        noteId: NoteId,
        attachmentId: AttachmentId,
    ): SuggestionOutcome = withNote(noteId) { note ->
        val attachment = note.attachments.firstOrNull { it.id == attachmentId }
            ?: return@withNote SuggestionOutcome.InvalidTarget("attachment does not belong to note")
        val source = attachmentContent.open(note.accountKey, attachmentId)
            ?: return@withNote SuggestionOutcome.InvalidTarget("attachment content is unavailable")
        val bytes = try {
            source.read()
        } catch (cancelled: CancellationException) {
            throw cancelled
        } catch (_: Exception) {
            return@withNote SuggestionOutcome.Failed("assistant.attachment_read_failed")
        }
        if (bytes.size.toLong() != attachment.sizeBytes) {
            return@withNote SuggestionOutcome.Failed("assistant.attachment_size_mismatch")
        }
        when (val outcome = images.tag(ImageInput.Bytes(bytes, attachment.mediaType))) {
            is ImageOutcome.Tagged -> ready(
                AssistantSuggestion.ImageTags(
                    noteId,
                    attachmentId,
                    outcome.tags.map { it.label.trim().lowercase() }
                        .filter(String::isNotBlank)
                        .map(::NoteTag)
                        .toSet(),
                ),
            )
            is ImageOutcome.NotConfigured -> SuggestionOutcome.Unavailable(outcome.reason)
            is ImageOutcome.Failed -> SuggestionOutcome.Failed(outcome.code)
        }
    }

    override suspend fun apply(suggestion: AssistantSuggestion): ApplySuggestionOutcome {
        val note = notes.get(suggestion.noteId) ?: return ApplySuggestionOutcome.NoteNotFound
        val command = when (suggestion) {
            is AssistantSuggestion.Summary -> UpdateNoteCommand(
                note.id,
                summary = FieldUpdate.Set(suggestion.value),
            )
            is AssistantSuggestion.TextTags -> UpdateNoteCommand(
                note.id,
                tags = FieldUpdate.Set(note.tags + suggestion.values),
            )
            is AssistantSuggestion.ImageTags -> {
                if (note.attachments.none { it.id == suggestion.attachmentId }) {
                    return ApplySuggestionOutcome.Rejected("attachment does not belong to note")
                }
                UpdateNoteCommand(note.id, tags = FieldUpdate.Set(note.tags + suggestion.values))
            }
            is AssistantSuggestion.Rewrite -> {
                var replaced = false
                val content = note.contentItems.map { item ->
                    if (item is ContentItem.Text && item.id == suggestion.contentItemId) {
                        replaced = true
                        item.copy(text = suggestion.value)
                    } else {
                        item
                    }
                }
                if (!replaced) return ApplySuggestionOutcome.Rejected("content item is not text")
                UpdateNoteCommand(note.id, contentItems = FieldUpdate.Set(content))
            }
        }
        return when (notes.update(command)) {
            is NoteMutationOutcome.Saved -> ApplySuggestionOutcome.Applied
            NoteMutationOutcome.NotFound -> ApplySuggestionOutcome.NoteNotFound
            NoteMutationOutcome.SignedOut -> ApplySuggestionOutcome.Rejected("signed out")
            is NoteMutationOutcome.Invalid -> ApplySuggestionOutcome.Rejected("invalid suggestion")
            NoteMutationOutcome.Deleted -> ApplySuggestionOutcome.Rejected("unexpected delete result")
        }
    }

    private suspend fun withNote(noteId: NoteId, block: suspend (Note) -> SuggestionOutcome): SuggestionOutcome =
        notes.get(noteId)?.let { block(it) } ?: SuggestionOutcome.NoteNotFound

    private fun ready(suggestion: AssistantSuggestion) = SuggestionOutcome.Ready(suggestion)
}

data class AssistantCoreComponent(val assistant: NoteAssistant) {
    companion object {
        fun create(
            notes: NotesService,
            attachmentContent: AttachmentContentPort,
            text: TextAssistant,
            images: ImageTagger,
        ): AssistantCoreComponent =
            AssistantCoreComponent(DefaultNoteAssistant(notes, attachmentContent, text, images))
    }
}

private fun Note.plainText(): String = contentItems.joinToString("\n") { item ->
    when (item) {
        is ContentItem.Text -> item.text
        is ContentItem.Image -> item.caption.orEmpty()
        is ContentItem.File -> ""
        is ContentItem.Link -> listOfNotNull(item.label, item.url).joinToString(" ")
    }
}

private fun AssistantRewriteStyle.toTextStyle(): RewriteStyle = when (this) {
    AssistantRewriteStyle.CONCISE -> RewriteStyle.CONCISE
    AssistantRewriteStyle.CLEAR -> RewriteStyle.CLEAR
    AssistantRewriteStyle.PROFESSIONAL -> RewriteStyle.PROFESSIONAL
    AssistantRewriteStyle.CASUAL -> RewriteStyle.CASUAL
}

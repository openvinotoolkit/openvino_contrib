// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package com.openvino.notes.assistant.core

import com.openvino.notes.ai.image.api.FakeImageTagger
import com.openvino.notes.ai.image.api.ImageOutcome
import com.openvino.notes.ai.image.api.ImageTag
import com.openvino.notes.ai.text.api.FakeTextAssistant
import com.openvino.notes.ai.text.api.RewriteOutcome
import com.openvino.notes.ai.text.api.SummaryOutcome
import com.openvino.notes.ai.text.api.TextTagsOutcome
import com.openvino.notes.assistant.api.ApplySuggestionOutcome
import com.openvino.notes.assistant.api.AssistantRewriteStyle
import com.openvino.notes.assistant.api.AssistantSuggestion
import com.openvino.notes.assistant.api.SuggestionOutcome
import com.openvino.notes.kernel.AccountKey
import com.openvino.notes.notes.api.AttachmentId
import com.openvino.notes.notes.api.AttachmentMetadata
import com.openvino.notes.notes.api.BinarySource
import com.openvino.notes.notes.api.ContentItem
import com.openvino.notes.notes.api.ContentItemId
import com.openvino.notes.notes.api.FakeNotesService
import com.openvino.notes.notes.api.FakeAttachmentContentPort
import com.openvino.notes.notes.api.Note
import com.openvino.notes.notes.api.NoteId
import com.openvino.notes.notes.api.NoteTag
import java.time.Instant
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class DefaultNoteAssistantTest {
    @Test fun `summary is applied to summary without changing content`() = runTest {
        val notes = FakeNotesService(listOf(note()))
        val text = FakeTextAssistant().apply { summaryOutcome = SummaryOutcome.Generated("Short summary") }
        val assistant = DefaultNoteAssistant(notes, FakeAttachmentContentPort(), text, FakeImageTagger())

        val suggestion = (assistant.summarize(NoteId("note")) as SuggestionOutcome.Ready).suggestion

        assertEquals(ApplySuggestionOutcome.Applied, assistant.apply(suggestion))
        assertEquals("Short summary", notes.get(NoteId("note"))?.summary)
        assertEquals("Long body", (notes.get(NoteId("note"))?.contentItems?.single() as ContentItem.Text).text)
    }

    @Test fun `text tags are normalized and merged without duplicates`() = runTest {
        val notes = FakeNotesService(listOf(note().copy(tags = setOf(NoteTag("existing")))))
        val text = FakeTextAssistant().apply {
            tagsOutcome = TextTagsOutcome.Generated(setOf("existing", "new"))
        }
        val assistant = DefaultNoteAssistant(notes, FakeAttachmentContentPort(), text, FakeImageTagger())

        val suggestion = (assistant.suggestTextTags(NoteId("note")) as SuggestionOutcome.Ready).suggestion
        assistant.apply(suggestion)

        assertEquals(setOf(NoteTag("existing"), NoteTag("new")), notes.get(NoteId("note"))?.tags)
    }

    @Test fun `rewrite replaces only the targeted text item`() = runTest {
        val original = note().copy(isFavorite = true, summary = "Keep")
        val notes = FakeNotesService(listOf(original))
        val text = FakeTextAssistant().apply { rewriteOutcome = RewriteOutcome.Generated("Clear body") }
        val assistant = DefaultNoteAssistant(notes, FakeAttachmentContentPort(), text, FakeImageTagger())

        val outcome = assistant.rewrite(NoteId("note"), ContentItemId("body"), AssistantRewriteStyle.CLEAR)
        assertTrue(outcome is SuggestionOutcome.Ready)
        assistant.apply((outcome as SuggestionOutcome.Ready).suggestion)

        val updated = notes.get(NoteId("note"))
        assertEquals("Clear body", (updated?.contentItems?.single() as ContentItem.Text).text)
        assertEquals(original.title, updated.title)
        assertEquals(original.isFavorite, updated.isFavorite)
        assertEquals(original.summary, updated.summary)
    }

    @Test fun `typed unavailable and failure outcomes are preserved`() = runTest {
        val text = FakeTextAssistant().apply {
            summaryOutcome = SummaryOutcome.NotConfigured("missing model")
            tagsOutcome = TextTagsOutcome.Failed("generation.failed")
        }
        val assistant = DefaultNoteAssistant(
            FakeNotesService(listOf(note())),
            FakeAttachmentContentPort(),
            text,
            FakeImageTagger(),
        )

        assertEquals(SuggestionOutcome.Unavailable("missing model"), assistant.summarize(NoteId("note")))
        assertEquals(SuggestionOutcome.Failed("generation.failed"), assistant.suggestTextTags(NoteId("note")))
    }

    @Test fun `image tags load attachment content through notes ownership`() = runTest {
        val noteId = NoteId("note")
        val attachmentId = AttachmentId("image")
        val contentItemId = ContentItemId("image-item")
        val bytes = byteArrayOf(1, 2, 3)
        val attachment = AttachmentMetadata(
            attachmentId,
            noteId,
            contentItemId,
            "image.png",
            "image/png",
            bytes.size.toLong(),
        )
        val target = note().copy(
            contentItems = listOf(ContentItem.Image(contentItemId, attachmentId)),
            attachments = listOf(attachment),
        )
        val content = FakeAttachmentContentPort().apply {
            put(target.accountKey, attachment, BinarySource { bytes })
        }
        val images = FakeImageTagger().apply {
            outcome = ImageOutcome.Tagged(listOf(ImageTag(" Diagram ", 0.9f)))
        }
        val assistant = DefaultNoteAssistant(FakeNotesService(listOf(target)), content, FakeTextAssistant(), images)

        val outcome = assistant.suggestImageTags(noteId, attachmentId) as SuggestionOutcome.Ready

        assertEquals(
            AssistantSuggestion.ImageTags(noteId, attachmentId, setOf(NoteTag("diagram"))),
            outcome.suggestion,
        )
    }

    private fun note() = Note(
        id = NoteId("note"),
        accountKey = AccountKey("account"),
        title = "Old",
        contentItems = listOf(ContentItem.Text(ContentItemId("body"), "Long body")),
        createdAt = Instant.EPOCH,
        updatedAt = Instant.EPOCH,
    )
}

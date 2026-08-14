// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package com.itlab.ai.text.openvino

import com.itlab.ai.text.api.RewriteOutcome
import com.itlab.ai.text.api.RewriteRequest
import com.itlab.ai.text.api.RewriteStyle
import com.itlab.ai.text.api.SummaryOutcome
import com.itlab.ai.text.api.SummaryRequest
import com.itlab.ai.text.api.TextTagsOutcome
import com.itlab.ai.text.api.TextTagsRequest
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class OpenVinoTextAssistantTest {
    @Test fun `pipelines own prompts and normalize operation-specific output`() = runTest {
        val prompts = mutableListOf<String>()
        val outputs = ArrayDeque(listOf(" Summary: concise ", "Tags: #One, one, - Two", "Answer: clearer"))
        val assistant = OpenVinoTextAssistant("model") { input, _ ->
            prompts += input
            outputs.removeFirst()
        }

        assertEquals(SummaryOutcome.Generated("concise"), assistant.summarize(SummaryRequest("private note")))
        assertEquals(TextTagsOutcome.Generated(setOf("one", "two")), assistant.suggestTags(TextTagsRequest("private note")))
        assertEquals(
            RewriteOutcome.Generated("clearer"),
            assistant.rewrite(RewriteRequest("private note", RewriteStyle.CLEAR)),
        )
        assertTrue(prompts.all { "CONTENT:\nprivate note" in it })
    }

    @Test fun `retry is bounded and cancellation is propagated`() = runTest {
        var attempts = 0
        val retrying = RetryingGenerator(TextGenerationBackend { _, _ ->
            attempts += 1
            throw TextGenerationException("temporary", retryable = true)
        })
        runCatching { retrying.generate("instruction", "content", 10) }
        assertEquals(2, attempts)

        val cancelling = RetryingGenerator(
            TextGenerationBackend { _, _ -> throw CancellationException("cancel") },
        )
        val failure = runCatching { cancelling.generate("instruction", "content", 10) }.exceptionOrNull()
        assertTrue(failure is CancellationException)
    }

    @Test fun `non retryable failures are mapped without retry`() = runTest {
        var attempts = 0
        val assistant = OpenVinoTextAssistant("model") { _, _ ->
            attempts += 1
            throw TextGenerationException("invalid_model", retryable = false)
        }

        assertEquals(SummaryOutcome.Failed("invalid_model"), assistant.summarize(SummaryRequest("note")))
        assertEquals(1, attempts)
    }
}

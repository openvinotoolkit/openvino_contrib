package com.itlab.ai

class ResultProcessor {
    fun normalizeSummary(raw: String): String = raw.trim()

    fun normalizeTags(raw: String): Set<String> =
        raw
            .split(',', '\n')
            .map { it.trim().lowercase() }
            .filter { it.isNotBlank() }
            .toSet()
}

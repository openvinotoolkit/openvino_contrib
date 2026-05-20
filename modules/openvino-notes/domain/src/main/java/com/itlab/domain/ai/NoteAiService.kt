package com.itlab.domain.ai

interface NoteAiService {
    suspend fun summarize(text: String): String

    suspend fun tagTXT(text: String): Set<String>

    suspend fun tagIMGs(img: List<String>): Set<String>
}

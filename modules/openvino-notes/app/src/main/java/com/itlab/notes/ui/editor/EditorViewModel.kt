package com.itlab.notes.ui.editor

import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import com.itlab.domain.model.ContentItem
import com.itlab.notes.media.withoutTextItems
import com.itlab.notes.ui.notes.NoteItemUi
import com.itlab.notes.ui.toSingleLineText

class EditorViewModel(
    initialNote: NoteItemUi,
) {
    private val noteId: String = initialNote.id
    private val userId: String = initialNote.userId
    private val folderId: String? = initialNote.folderId

    var title: String by mutableStateOf(initialNote.title.toSingleLineText())
        private set

    var content: String by mutableStateOf(initialNote.content)
        private set

    var attachments: List<ContentItem> by mutableStateOf(initialNote.attachments.withoutTextItems())
        private set

    var isFavorite: Boolean by mutableStateOf(initialNote.isFavorite)
        private set

    fun syncFavoriteFromNote(value: Boolean) {
        isFavorite = value
    }

    fun onTitleChange(newTitle: String) {
        title = newTitle.toSingleLineText()
    }

    fun onContentChange(newContent: String) {
        content = newContent
    }

    fun addAttachment(item: ContentItem) {
        attachments = attachments + item
    }

    fun addAttachments(items: List<ContentItem>) {
        if (items.isEmpty()) return
        attachments = attachments + items
    }

    fun removeAttachment(id: String) {
        attachments = attachments.filterNot { it.id == id }
    }

    /** Applies attachment list from DB (after save/sync); keeps only unsaved local imports not on server. */
    fun syncAttachmentsFromNote(fromNote: List<ContentItem>) {
        val incoming = fromNote.withoutTextItems()
        val incomingById = incoming.associateBy { it.id }
        val localById = attachments.associateBy { it.id }
        val merged =
            incoming.map { remote ->
                localById[remote.id]?.let { mergeItemSources(it, remote) } ?: remote
            }
        val unsavedLocal =
            attachments.filter { it.isUnsavedLocalAttachment() && it.id !in incomingById }
        attachments = merged + unsavedLocal
    }

    private fun mergeItemSources(
        local: ContentItem,
        remote: ContentItem,
    ): ContentItem =
        when (local) {
            is ContentItem.Image ->
                (remote as? ContentItem.Image)?.let { r ->
                    local.copy(
                        source =
                            local.source.copy(
                                localPath = local.source.localPath ?: r.source.localPath,
                                remoteUrl = local.source.remoteUrl ?: r.source.remoteUrl,
                            ),
                    )
                } ?: local
            is ContentItem.File ->
                (remote as? ContentItem.File)?.let { r ->
                    local.copy(
                        source =
                            local.source.copy(
                                localPath = local.source.localPath ?: r.source.localPath,
                                remoteUrl = local.source.remoteUrl ?: r.source.remoteUrl,
                            ),
                    )
                } ?: local
            else -> local
        }

    private fun ContentItem.isUnsavedLocalAttachment(): Boolean {
        val source =
            when (this) {
                is ContentItem.Image -> this.source
                is ContentItem.File -> this.source
                else -> return false
            }
        return !source.localPath.isNullOrBlank() && source.remoteUrl.isNullOrBlank()
    }

    fun buildUpdatedNote(): NoteItemUi =
        NoteItemUi(
            id = noteId,
            userId = userId,
            title = title,
            content = content,
            folderId = folderId,
            attachments = attachments,
            isFavorite = isFavorite,
        )
}

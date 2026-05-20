package com.itlab.notes.media

import android.content.Context
import android.net.Uri
import android.webkit.MimeTypeMap
import com.itlab.domain.model.ContentItem
import com.itlab.domain.model.DataSource
import java.io.File
import java.util.UUID

object NoteMediaImport {
    private const val SUBDIR = "note_attachments"

    fun importImagesFromUris(
        context: Context,
        uris: List<Uri>,
    ): List<ContentItem.Image> =
        uris.mapNotNull { uri ->
            runCatching { importImageFromUri(context, uri) }.getOrNull()
        }

    fun importImageFromUri(
        context: Context,
        uri: Uri,
    ): ContentItem.Image {
        val appContext = context.applicationContext
        val resolver = appContext.contentResolver
        val mime = resolver.getType(uri) ?: "image/jpeg"
        val ext = MimeTypeMap.getSingleton().getExtensionFromMimeType(mime) ?: "jpg"
        val dir = File(appContext.filesDir, SUBDIR).apply { mkdirs() }
        val file = File(dir, "${UUID.randomUUID()}.$ext")
        resolver.openInputStream(uri)?.use { input ->
            file.outputStream().use { out -> input.copyTo(out) }
        } ?: error("Cannot read selected image")
        require(file.length() > 0L) { "Selected image is empty" }
        return ContentItem.Image(
            source = DataSource(localPath = file.absolutePath),
            mimeType = mime,
        )
    }

    fun deleteImportedFileIfOwned(
        context: Context,
        localPath: String?,
    ) {
        val path = localPath ?: return
        val file = File(path)
        if (!file.exists()) return
        val root = File(context.applicationContext.filesDir, SUBDIR).absolutePath
        if (file.absolutePath.startsWith(root)) {
            file.delete()
        }
    }
}

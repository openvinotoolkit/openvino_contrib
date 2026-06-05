package com.itlab.notes.media

import android.content.Context
import com.itlab.domain.model.ContentItem
import com.itlab.domain.model.DataSource
import java.io.File

fun List<ContentItem>.withoutTextItems(): List<ContentItem> = filterNot { it is ContentItem.Text }

fun List<ContentItem>.imageAttachments(): List<ContentItem.Image> = filterIsInstance<ContentItem.Image>()

private fun readableFile(path: String?): File? {
    if (path.isNullOrBlank()) return null
    val file = File(path)
    return file.takeIf { it.isFile && it.canRead() && it.length() > 0L }
}

/** Local file written by cloud sync: `files/media/{mediaId}`. */
fun syncedMediaFile(
    context: Context,
    mediaId: String,
): File? = readableFile(File(context.applicationContext.filesDir, "media/$mediaId").absolutePath)

fun DataSource.toCoilModel(context: Context): Any? {
    readableFile(localPath)?.let { return it }
    remoteUrl
        ?.takeIf { it.startsWith("http://", ignoreCase = true) || it.startsWith("https://", ignoreCase = true) }
        ?.let { return it }
    return null
}

fun ContentItem.Image.toCoilModel(context: Context): Any? {
    source.toCoilModel(context)?.let { return it }
    syncedMediaFile(context, id)?.let { return it }
    return null
}

fun ContentItem.File.toCoilModel(context: Context): Any? {
    source.toCoilModel(context)?.let { return it }
    syncedMediaFile(context, id)?.let { return it }
    return null
}

/** True while cloud pull may still be writing the local file for this attachment. */
fun ContentItem.Image.isMediaLoadPending(
    context: Context,
    cloudSyncInProgress: Boolean,
): Boolean = toCoilModel(context) == null && cloudSyncInProgress

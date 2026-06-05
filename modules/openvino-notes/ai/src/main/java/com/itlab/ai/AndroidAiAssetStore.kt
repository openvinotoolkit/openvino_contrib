package com.itlab.ai

import android.content.Context
import java.io.File
import java.io.IOException

internal class AndroidAiAssetStore(
    private val context: Context,
) {
    fun directoryExists(assetPath: String): Boolean =
        runCatching {
            context.assets
                .list(assetPath)
                .orEmpty()
                .isNotEmpty()
        }.getOrDefault(false)

    fun readTextOrNull(assetPath: String): String? =
        runCatching {
            context.assets
                .open(assetPath)
                .bufferedReader()
                .use { it.readText() }
        }.getOrNull()

    @Throws(IOException::class)
    fun copyDirectory(
        assetPath: String,
        targetDir: File,
    ) {
        val entries = context.assets.list(assetPath).orEmpty()
        if (entries.isEmpty()) {
            throw IOException("Asset directory is missing or empty: $assetPath")
        }

        targetDir.mkdirs()
        entries.forEach { entry ->
            val childAssetPath = "$assetPath/$entry"
            val childTarget = File(targetDir, entry)
            val childEntries = context.assets.list(childAssetPath).orEmpty()
            if (childEntries.isEmpty()) {
                copyFile(childAssetPath, childTarget)
            } else {
                copyDirectory(childAssetPath, childTarget)
            }
        }
    }

    @Throws(IOException::class)
    fun copyFile(
        assetPath: String,
        targetFile: File,
    ) {
        targetFile.parentFile?.mkdirs()
        context.assets.open(assetPath).use { input ->
            targetFile.outputStream().use { output ->
                input.copyTo(output)
            }
        }
    }
}

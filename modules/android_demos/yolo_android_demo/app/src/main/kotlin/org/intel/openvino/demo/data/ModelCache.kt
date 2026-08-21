// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package org.intel.openvino.demo.data

import android.content.Context
import java.io.File

/**
 * On-disk cache for downloaded ONNX models, under the app's private files dir. A model is
 * "present" only if the file exists AND its size matches the manifest's expected [ModelEntry.bytes]
 * (a partial/interrupted download is treated as absent so it will be re-fetched).
 */
class ModelCache(context: Context) {

    private val dir: File = File(context.filesDir, "models").apply { mkdirs() }

    fun fileFor(entry: ModelEntry): File = File(dir, entry.fileName)

    fun isPresent(entry: ModelEntry): Boolean {
        val f = fileFor(entry)
        return f.isFile && f.length() == entry.bytes
    }

    /** Absolute path if cached, else null. */
    fun pathIfPresent(entry: ModelEntry): String? =
        if (isPresent(entry)) fileFor(entry).absolutePath else null

    fun delete(entry: ModelEntry) {
        fileFor(entry).delete()
    }

    /** Total bytes currently used by cached model files. */
    fun totalBytes(): Long =
        dir.listFiles()?.filter { it.isFile }?.sumOf { it.length() } ?: 0L

    /** Number of cached model files. */
    fun count(): Int = dir.listFiles()?.count { it.isFile } ?: 0

    /** Delete every cached model file; returns the number of bytes freed. */
    fun clearAll(): Long {
        val files = dir.listFiles()?.filter { it.isFile } ?: return 0L
        var freed = 0L
        for (f in files) {
            val len = f.length()
            if (f.delete()) freed += len
        }
        return freed
    }
}

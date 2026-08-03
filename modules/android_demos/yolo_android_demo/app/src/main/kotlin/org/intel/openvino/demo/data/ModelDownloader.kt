// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package org.intel.openvino.demo.data

import android.content.Context
import android.net.ConnectivityManager
import android.net.NetworkCapabilities
import android.util.Log
import java.io.File
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Streams a model's ONNX from its manifest URL (Ultralytics' official release asset by default)
 * into the on-disk cache, reporting progress, supporting cancel, and verifying the sha256 on
 * completion. This is the **only** network I/O in the app, and it uses the built-in
 * [HttpURLConnection].
 *
 * Not tied to any Activity: run it off a coroutine/executor so it survives configuration changes.
 */
class ModelDownloader(
    private val context: Context,
    private val cache: ModelCache,
) {
    companion object {
        private const val TAG = "ModelDownloader"
        private const val BUFFER = 64 * 1024
        private const val CONNECT_TIMEOUT_MS = 15_000
        private const val READ_TIMEOUT_MS = 30_000
    }

    private val cancelled = AtomicBoolean(false)

    fun cancel() { cancelled.set(true) }

    /** True if the device currently has a validated internet-capable network. */
    fun hasInternet(): Boolean {
        val cm = context.getSystemService(Context.CONNECTIVITY_SERVICE) as? ConnectivityManager
            ?: return false
        val net = cm.activeNetwork ?: return false
        val caps = cm.getNetworkCapabilities(net) ?: return false
        return caps.hasCapability(NetworkCapabilities.NET_CAPABILITY_INTERNET) &&
            caps.hasCapability(NetworkCapabilities.NET_CAPABILITY_VALIDATED)
    }

    /**
     * Download [entry] to the cache, invoking [onProgress] with (percent, downloaded, total).
     * Returns a terminal [DownloadState] (Done/Error). Blocking — call off the main thread.
     * A cache hit returns Done immediately without touching the network.
     */
    fun download(entry: ModelEntry, onProgress: (Int, Long, Long) -> Unit): DownloadState {
        cache.pathIfPresent(entry)?.let {
            Log.i(TAG, "Cache hit for ${entry.id}: $it")
            return DownloadState.Done(it)
        }
        if (!hasInternet()) {
            return DownloadState.Error(DownloadErrorKind.NO_INTERNET, "No internet connection.")
        }

        cancelled.set(false)
        val dest = cache.fileFor(entry)
        val tmp = File(dest.absolutePath + ".part")
        var conn: HttpURLConnection? = null
        try {
            var url = URL(entry.url)
            conn = openFollowingRedirects(url)
            val code = conn.responseCode
            if (code !in 200..299) {
                return DownloadState.Error(
                    DownloadErrorKind.NETWORK, "HTTP $code fetching ${entry.url}",
                )
            }
            val total = if (entry.bytes > 0) entry.bytes else conn.contentLengthLong

            val digest = MessageDigest.getInstance("SHA-256")
            conn.inputStream.use { input ->
                tmp.outputStream().use { output ->
                    val buf = ByteArray(BUFFER)
                    var downloaded = 0L
                    var lastPct = -1
                    while (true) {
                        if (cancelled.get()) {
                            tmp.delete()
                            return DownloadState.Error(DownloadErrorKind.CANCELLED, "Cancelled")
                        }
                        val n = input.read(buf)
                        if (n < 0) break
                        output.write(buf, 0, n)
                        digest.update(buf, 0, n)
                        downloaded += n
                        if (total > 0) {
                            val pct = ((downloaded * 100) / total).toInt().coerceIn(0, 100)
                            if (pct != lastPct) {
                                lastPct = pct
                                onProgress(pct, downloaded, total)
                            }
                        }
                    }
                }
            }

            // Verify integrity before accepting the file.
            val hex = digest.digest().joinToString("") { "%02x".format(it) }
            if (entry.sha256.isNotBlank() && !hex.equals(entry.sha256, ignoreCase = true)) {
                tmp.delete()
                Log.e(TAG, "Checksum mismatch for ${entry.id}: got $hex want ${entry.sha256}")
                return DownloadState.Error(
                    DownloadErrorKind.CHECKSUM, "Downloaded file failed integrity check.",
                )
            }

            if (!tmp.renameTo(dest)) {
                tmp.copyTo(dest, overwrite = true); tmp.delete()
            }
            Log.i(TAG, "Downloaded ${entry.id} -> ${dest.absolutePath} (${dest.length()} bytes)")
            return DownloadState.Done(dest.absolutePath)
        } catch (t: Throwable) {
            tmp.delete()
            Log.e(TAG, "Download failed for ${entry.id}", t)
            val kind = if (!hasInternet()) DownloadErrorKind.NO_INTERNET else DownloadErrorKind.NETWORK
            return DownloadState.Error(kind, t.message ?: "Download failed")
        } finally {
            conn?.disconnect()
        }
    }

    /** Open a connection, manually following up to 5 redirects (incl. http<->https, which the JDK won't). */
    private fun openFollowingRedirects(start: URL): HttpURLConnection {
        var url = start
        var redirects = 0
        while (true) {
            val c = (url.openConnection() as HttpURLConnection).apply {
                connectTimeout = CONNECT_TIMEOUT_MS
                readTimeout = READ_TIMEOUT_MS
                instanceFollowRedirects = true
                setRequestProperty("User-Agent", "openvino-yolo-demo")
            }
            val code = c.responseCode
            if (code in listOf(301, 302, 303, 307, 308) && redirects < 5) {
                val loc = c.getHeaderField("Location")
                c.disconnect()
                url = URL(url, loc)
                redirects++
                continue
            }
            return c
        }
    }
}

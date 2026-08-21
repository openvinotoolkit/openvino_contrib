// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package org.intel.openvino.demo.ui

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.LiveData
import org.intel.openvino.demo.data.DownloadState
import org.intel.openvino.demo.data.ModelCache
import org.intel.openvino.demo.data.ModelDownloader
import org.intel.openvino.demo.data.ModelEntry
import java.util.concurrent.Executors

/**
 * Owns the model download so it survives configuration changes (rotation, backgrounding). Runs the
 * blocking [ModelDownloader] on a background executor and publishes [DownloadState] via LiveData.
 */
class DownloadViewModel(app: Application) : AndroidViewModel(app) {

    private val cache = ModelCache(app)
    private val downloader = ModelDownloader(app, cache)
    private val executor = Executors.newSingleThreadExecutor()

    private val _state = MutableLiveData<DownloadState>(DownloadState.Idle)
    val state: LiveData<DownloadState> = _state

    @Volatile private var active = false

    /** Path if the model is already cached (so the UI can show "Ready — no download"). */
    fun cachedPath(entry: ModelEntry): String? = cache.pathIfPresent(entry)

    /** Total bytes used by cached models, for a "Clear cache (N MB)" label. */
    fun cacheBytes(): Long = cache.totalBytes()

    fun cacheCount(): Int = cache.count()

    /** Delete all cached models; returns bytes freed. */
    fun clearCache(): Long = cache.clearAll()

    fun hasInternet(): Boolean = downloader.hasInternet()

    /** Start (or restart) downloading [entry]. Idempotent while one is already running. */
    fun start(entry: ModelEntry) {
        if (active) return
        active = true
        _state.postValue(DownloadState.Downloading(0, 0, entry.bytes))
        executor.execute {
            val result = downloader.download(entry) { pct, done, total ->
                _state.postValue(DownloadState.Downloading(pct, done, total))
            }
            active = false
            _state.postValue(result)
        }
    }

    fun cancel() {
        downloader.cancel()
    }

    fun reset() {
        _state.postValue(DownloadState.Idle)
    }

    override fun onCleared() {
        downloader.cancel()
        executor.shutdownNow()
    }
}

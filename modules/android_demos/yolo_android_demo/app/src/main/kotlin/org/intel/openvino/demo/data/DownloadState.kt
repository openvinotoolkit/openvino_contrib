// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package org.intel.openvino.demo.data

/** Why a model download failed, so the UI can show an actionable message. */
enum class DownloadErrorKind {
    NO_INTERNET,   // device has no connectivity
    NETWORK,       // connection dropped / HTTP error mid-transfer
    CHECKSUM,      // sha256 mismatch — corrupted or tampered file (deleted)
    IO,            // local storage error
    CANCELLED,     // user cancelled
}

/** Explicit states for the selector → download → camera flow, safe across config changes. */
sealed class DownloadState {
    /** Nothing in progress. */
    object Idle : DownloadState()

    /** Streaming the model; [percent] 0..100, [downloadedBytes]/[totalBytes] for a byte label. */
    data class Downloading(
        val percent: Int,
        val downloadedBytes: Long,
        val totalBytes: Long,
    ) : DownloadState()

    /** Model is ready on disk at [path] (fresh download or cache hit). */
    data class Done(val path: String) : DownloadState()

    /** Download failed; [kind] drives the dialog message, [message] adds detail. */
    data class Error(val kind: DownloadErrorKind, val message: String) : DownloadState()
}

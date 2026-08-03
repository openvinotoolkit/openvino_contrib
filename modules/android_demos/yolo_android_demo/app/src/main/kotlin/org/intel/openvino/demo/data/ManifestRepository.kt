// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package org.intel.openvino.demo.data

import android.content.Context
import android.util.Log

/**
 * Loads the static model manifest bundled in the APK's `assets/` — no network. The manifest lists
 * Ultralytics' official ONNX release-asset URLs plus metadata (§6a); the picker uses it offline.
 *
 * An optional base-host override (via [BuildConfig]/settings) lets an advanced user point the
 * download at a mirror or LAN server; the default requires no hosting by anyone.
 */
class ManifestRepository(private val context: Context) {

    companion object {
        private const val TAG = "ManifestRepository"
        private const val MANIFEST_ASSET = "models_manifest.json"
    }

    fun load(): ModelManifest {
        val json = context.assets.open(MANIFEST_ASSET).bufferedReader().use { it.readText() }
        val manifest = ModelManifest.parse(json)
        Log.i(TAG, "Loaded manifest ${manifest.releaseTags} with ${manifest.models.size} models")
        return applyBaseUrlOverride(manifest)
    }

    /**
     * If an override base host is configured, rewrite each entry's URL host while keeping the file
     * path (so a mirror only needs the same directory layout). No override → URLs unchanged.
     */
    private fun applyBaseUrlOverride(manifest: ModelManifest): ModelManifest {
        val override = org.intel.openvino.demo.BuildConfig.MODEL_BASE_URL_OVERRIDE
        if (override.isBlank()) return manifest
        val base = override.trimEnd('/')
        val rewritten = manifest.models.map { e ->
            val fileName = e.url.substringAfterLast('/')
            e.copy(url = "$base/$fileName")
        }
        Log.i(TAG, "Applied model base-URL override: $base")
        return manifest.copy(models = rewritten)
    }
}

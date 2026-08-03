// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package org.intel.openvino.demo.data

import org.json.JSONObject

/**
 * One model entry from the bundled `assets/models_manifest.json`. The manifest is a static list of
 * hyperlinks + metadata pointing at Ultralytics' **official** pre-exported ONNX release assets — it
 * is not the model weights, so it is distribution-safe. It is the single source of truth for the
 * on-device picker; no model metadata is hardcoded in Kotlin.
 */
data class ModelEntry(
    val id: String,          // e.g. "yolov8n-seg"
    val version: String,     // "v8" | "v11"
    val size: String,        // "n" | "s" | "m" | "l" | "x"
    val task: String,        // "detect" | "segment" | "pose" | "classify"
    val url: String,         // official Ultralytics .onnx release asset URL
    val bytes: Long,         // exact size, drives the UI label + progress bar
    val sha256: String,      // integrity check after download
    val imgsz: Int,          // square input size (e.g. 640, or 224 for classify)
    val layout: String,      // "NCHW"
    val scale: Float,        // normalization divisor (255)
    val labels: String,      // key into a bundled labels file ("coco80" / "imagenet1k" / ...)
    val variant: String,     // "anchor-free"
    /**
     * True if the ONNX has a dynamic spatial input ([1,3,H,W] with H/W free). For these, the app
     * reshapes the model to a size derived from the camera frame's aspect ratio (rounded to a
     * multiple of 32), so inference runs on a near-frame-shaped input with minimal padding. For
     * static models, [imgsz] is used as a fixed square.
     */
    val dynamic: Boolean,
) {
    /** Human-readable name for the picker, e.g. "YOLOv8 Nano · Segment". */
    val displayName: String
        get() = "${versionLabel(version)} ${sizeLabel(size)} · ${task.replaceFirstChar { it.uppercase() }}"

    /** Local filename to cache the downloaded ONNX under. */
    val fileName: String get() = "$id.onnx"

    companion object {
        /** Full version label: v8 -> YOLOv8, v10 -> YOLOv10, v11 -> YOLO11. */
        fun versionLabel(version: String): String = when (version) {
            "v8" -> "YOLOv8"
            "v10" -> "YOLOv10"
            "v11" -> "YOLO11"
            else -> "YOLO${version.removePrefix("v")}"
        }

        /** Readable size names instead of single letters (n/s/m/b/l/x). */
        fun sizeLabel(size: String): String = when (size.lowercase()) {
            "n" -> "Nano"
            "s" -> "Small"
            "m" -> "Medium"
            "b" -> "Balanced"      // YOLOv10-only size
            "l" -> "Large"
            "x" -> "Extra-Large"
            else -> size.uppercase()
        }

        fun fromJson(o: JSONObject): ModelEntry {
            val input = o.getJSONObject("input")
            val norm = o.getJSONObject("normalization")
            return ModelEntry(
                id = o.getString("id"),
                version = o.getString("version"),
                size = o.getString("size"),
                task = o.getString("task"),
                url = o.getString("url"),
                bytes = o.getLong("bytes"),
                sha256 = o.getString("sha256"),
                imgsz = input.getInt("imgsz"),
                layout = input.optString("layout", "NCHW"),
                scale = norm.optDouble("scale", 255.0).toFloat(),
                labels = o.optString("labels", "coco80"),
                variant = o.optString("variant", "anchor-free"),
                dynamic = o.optBoolean("dynamic", false),
            )
        }
    }
}

/** The parsed manifest: the release tags it was authored from plus every model entry. */
data class ModelManifest(
    val releaseTags: String,
    val models: List<ModelEntry>,
) {
    val versions: List<String> get() = models.map { it.version }.distinct()

    fun sizesFor(version: String): List<String> =
        models.filter { it.version == version }.map { it.size }.distinct()

    fun tasksFor(version: String, size: String): List<String> =
        models.filter { it.version == version && it.size == size }.map { it.task }.distinct()

    fun find(version: String, size: String, task: String): ModelEntry? =
        models.firstOrNull { it.version == version && it.size == size && it.task == task }

    companion object {
        fun parse(json: String): ModelManifest {
            val root = JSONObject(json)
            val arr = root.getJSONArray("models")
            val models = (0 until arr.length()).map { ModelEntry.fromJson(arr.getJSONObject(it)) }
            // Accept both the combined "releaseTags" object and a legacy "releaseTag" string.
            val tags = root.opt("releaseTags")?.toString() ?: root.optString("releaseTag", "")
            return ModelManifest(tags, models)
        }
    }
}

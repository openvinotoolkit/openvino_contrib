// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package org.intel.openvino.demo.camera

import android.content.Context
import android.util.Log
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

/**
 * Wraps CameraX setup: a [Preview] bound to a [PreviewView] plus an [ImageAnalysis] use case that
 * delivers YUV_420_888 frames to [onFrame] on a dedicated single-threaded executor — so inference
 * never runs on the UI thread. Backpressure is [ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST], so a slow
 * analyzer only ever drops frames; the preview stays smooth.
 */
class CameraController(
    private val context: Context,
    private val lifecycleOwner: LifecycleOwner,
    private val previewView: PreviewView,
) {
    companion object {
        private const val TAG = "CameraController"
    }

    /** Single inference thread: the analyzer callback runs here, one frame at a time. */
    val analysisExecutor: ExecutorService = Executors.newSingleThreadExecutor()

    private var cameraProvider: ProcessCameraProvider? = null

    /**
     * Start the camera. [onFrame] is invoked on [analysisExecutor] for the latest frame; the caller
     * MUST call `image.close()` when done (this class does it after the callback returns).
     */
    fun start(onFrame: (ImageProxy) -> Unit) {
        val future = ProcessCameraProvider.getInstance(context)
        future.addListener({
            val provider = future.get()
            cameraProvider = provider

            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }

            val analysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
                .build()

            analysis.setAnalyzer(analysisExecutor) { image ->
                try {
                    onFrame(image)
                } catch (t: Throwable) {
                    Log.e(TAG, "Frame analysis failed", t)
                } finally {
                    image.close()
                }
            }

            try {
                provider.unbindAll()
                provider.bindToLifecycle(
                    lifecycleOwner,
                    CameraSelector.DEFAULT_BACK_CAMERA,
                    preview,
                    analysis,
                )
            } catch (t: Throwable) {
                Log.e(TAG, "Failed to bind camera use cases", t)
            }
        }, ContextCompat.getMainExecutor(context))
    }

    fun stop() {
        cameraProvider?.unbindAll()
        analysisExecutor.shutdown()
    }
}

// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package org.intel.openvino.demo.ui

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.ImageProxy
import androidx.camera.view.PreviewView
import android.widget.Toast
import org.intel.openvino.Core
import org.intel.openvino.demo.R
import org.intel.openvino.demo.camera.CameraController
import org.intel.openvino.demo.data.ManifestRepository
import org.intel.openvino.demo.inference.ModelConfig
import org.intel.openvino.demo.inference.OvEngine
import org.intel.openvino.demo.postprocess.InferenceResult
import org.intel.openvino.demo.postprocess.YoloDecoder
import org.intel.openvino.demo.preprocess.FramePreprocessor

/**
 * Live screen (§6b): opened only after a model's ONNX is present on disk. Builds the [ModelConfig]
 * from the manifest entry + bundled labels, loads the ONNX in [OvEngine], and runs CameraX preview +
 * inference + overlay. The camera/preprocess/inference/overlay layers are YOLO-agnostic; all
 * task specifics live behind the [YoloDecoder] chosen from the config's task.
 */
class CameraActivity : AppCompatActivity() {

    companion object {
        const val EXTRA_MODEL_ID = "model_id"
        const val EXTRA_ONNX_PATH = "onnx_path"
        private const val TAG = "YoloDemo"
        private const val CONF_THRESHOLD = 0.25f
        private const val IOU_THRESHOLD = 0.45f
    }

    private lateinit var previewView: PreviewView
    private lateinit var overlayView: OverlayView
    private lateinit var loadingGroup: android.view.View

    private lateinit var core: Core
    private lateinit var engine: OvEngine
    private lateinit var config: ModelConfig
    private lateinit var decoder: YoloDecoder
    private lateinit var preprocessor: FramePreprocessor
    private var labels: List<String> = emptyList()

    private var cameraController: CameraController? = null

    @Volatile private var emaFps: Double = 0.0
    @Volatile private var compileStarted = false

    private val requestPermission =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) startCamera()
            else Toast.makeText(this, R.string.camera_permission_rationale, Toast.LENGTH_LONG).show()
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_camera)
        previewView = findViewById(R.id.previewView)
        overlayView = findViewById(R.id.overlayView)
        loadingGroup = findViewById(R.id.loadingGroup)

        val modelId = intent.getStringExtra(EXTRA_MODEL_ID)
        val onnxPath = intent.getStringExtra(EXTRA_ONNX_PATH)
        if (modelId == null || onnxPath == null) {
            Toast.makeText(this, "No model specified", Toast.LENGTH_LONG).show(); finish(); return
        }

        // Build the config from the manifest entry + bundled labels.
        val manifest = ManifestRepository(this).load()
        val entry = manifest.models.firstOrNull { it.id == modelId }
        if (entry == null) {
            Toast.makeText(this, "Unknown model $modelId", Toast.LENGTH_LONG).show(); finish(); return
        }
        modelEntry = entry
        modelOnnxPath = onnxPath
        labels = loadLabels(entry.labels)
        title = entry.displayName

        // A static model can be compiled immediately (fixed square input). A dynamic model needs the
        // camera frame's dimensions to pick a frame-shaped input, so its compile is deferred to the
        // first frame (or the test image). Show the "Compiling model…" indicator until ready.
        loadingGroup.visibility = android.view.View.VISIBLE
        core = Core()
        engine = OvEngine(core)

        if (!entry.dynamic) {
            compileAsync(frameW = 0, frameH = 0) { onModelReady() }
        } else {
            // Defer: start the camera (or test image); compile once we know the frame size.
            onModelReady()
        }
    }

    /** Store the entry until we can build the config (dynamic models need frame dims first). */
    private lateinit var modelEntry: org.intel.openvino.demo.data.ModelEntry
    private lateinit var modelOnnxPath: String
    @Volatile private var modelReady = false

    /** Build the config for the given (upright) frame size, compile off-thread, then run [onDone]. */
    private fun compileAsync(frameW: Int, frameH: Int, onDone: () -> Unit) {
        config = ModelConfig.from(modelEntry, modelOnnxPath, labels, frameW, frameH)
        decoder = YoloDecoder.forConfig(config)
        preprocessor = FramePreprocessor(config.inputWidth, config.inputHeight)
        runOnUiThread { loadingGroup.visibility = android.view.View.VISIBLE }
        Thread {
            val t0 = System.nanoTime()
            val ok = try { engine.load(config); true } catch (t: Throwable) {
                Log.e(TAG, "Failed to load/compile model", t); false
            }
            val ms = (System.nanoTime() - t0) / 1_000_000
            runOnUiThread {
                loadingGroup.visibility = android.view.View.GONE
                if (!ok) { Toast.makeText(this, "Model compile failed", Toast.LENGTH_LONG).show(); finish(); return@runOnUiThread }
                modelReady = true
                Log.i(TAG, "Model compiled in ${ms}ms: ${config.id} input=${config.inputWidth}x${config.inputHeight} dynamic=${config.dynamic}")
                onDone()
            }
        }.apply { name = "ov-compile" }.start()
    }

    /** Called once the (static) model is compiled, or immediately for dynamic to begin the camera. */
    private fun onModelReady() {
        // Static-image test mode (no camera): --es source test [reuses the same pipeline].
        if (intent.getStringExtra("source") == "test") { runStaticImageTest(); return }

        if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            startCamera()
        } else {
            requestPermission.launch(Manifest.permission.CAMERA)
        }
    }

    private fun startCamera() {
        val controller = CameraController(this, this, previewView)
        cameraController = controller
        controller.start { image -> onFrame(image, image.imageInfo.rotationDegrees) }
    }

    /** Runs on the CameraX analysis (inference) thread. */
    private fun onFrame(image: ImageProxy, rotation: Int) {
        // Dynamic model: on the first frame, compile for this frame's upright size, then skip until ready.
        if (!modelReady) {
            if (modelEntry.dynamic) {
                val fw = if (rotation == 90 || rotation == 270) image.height else image.width
                val fh = if (rotation == 90 || rotation == 270) image.width else image.height
                if (!compileStarted) { compileStarted = true; compileAsync(fw, fh) {} }
            }
            return // drop frames until the model is compiled
        }
        val start = System.nanoTime()
        val rgb = preprocessor.process(image, rotation)
        val outputs = engine.infer(rgb)
        val result = decoder.decode(outputs, preprocessor.lastTransform, config, config.labels, CONF_THRESHOLD, IOU_THRESHOLD)
        val ms = (System.nanoTime() - start) / 1_000_000
        val instFps = if (ms > 0) 1000.0 / ms else 0.0
        emaFps = if (emaFps == 0.0) instFps else 0.9 * emaFps + 0.1 * instFps
        val t = preprocessor.lastTransform
        Log.i(TAG, "infer ${config.id} ${ms}ms det=${result.detections.size} masks=${result.masks.size} poses=${result.poses.size} cls=${result.classifications.size}")
        overlayView.setResult(result, t.frameWidth, t.frameHeight, "%.1f FPS | %d ms | %s".format(emaFps, ms, config.displayName))
    }

    /**
     * Camera-free verification path (developer tool): run a still image through the same
     * preprocess → infer → decode → render pipeline, with no camera. The image is supplied by the
     * caller via `--es image_path <file>` (e.g. adb-pushed to /data/local/tmp). No test image is
     * bundled in the app, so nothing is redistributed. Example:
     *   adb shell am start -n .../.ui.CameraActivity --es model_id yolov8n --es onnx_path <onnx> \
     *       --es source test --es image_path /data/local/tmp/sample.jpg
     */
    private fun runStaticImageTest() {
        val imagePath = intent.getStringExtra("image_path")
        if (imagePath == null) {
            Log.e(TAG, "test mode requires --es image_path <file>"); finish(); return
        }
        java.util.concurrent.Executors.newSingleThreadExecutor().execute {
            val bmp = try {
                android.graphics.BitmapFactory.decodeFile(imagePath)
                    ?: throw java.io.IOException("could not decode $imagePath")
            } catch (t: Throwable) { Log.e(TAG, "test image load failed", t); return@execute }
            // Dynamic model: compile for the test image's dimensions first (blocks until ready).
            if (!modelReady) {
                config = ModelConfig.from(modelEntry, modelOnnxPath, labels, bmp.width, bmp.height)
                decoder = YoloDecoder.forConfig(config)
                preprocessor = FramePreprocessor(config.inputWidth, config.inputHeight)
                try { engine.load(config); modelReady = true } catch (t: Throwable) {
                    Log.e(TAG, "TEST compile failed", t); return@execute
                }
                runOnUiThread { loadingGroup.visibility = android.view.View.GONE }
                Log.i(TAG, "TEST compiled ${config.id} input=${config.inputWidth}x${config.inputHeight} dynamic=${config.dynamic}")
            }
            val start = System.nanoTime()
            val rgb = preprocessor.processBitmap(bmp)
            val outputs = engine.infer(rgb)
            val result = decoder.decode(outputs, preprocessor.lastTransform, config, config.labels, CONF_THRESHOLD, IOU_THRESHOLD)
            val ms = (System.nanoTime() - start) / 1_000_000
            Log.i(TAG, "TEST ${config.id} task=${config.task} ${ms}ms det=${result.detections.size} masks=${result.masks.size} poses=${result.poses.size} cls=${result.classifications.size} " +
                "top=${result.detections.take(3).map { "${it.label}:${"%.2f".format(it.score)}" }}${result.classifications.take(3).map { "${it.label}:${"%.2f".format(it.score)}" }}")
            overlayView.setStaticResult(bmp, result, "TEST ${config.displayName} | ${ms}ms")
        }
    }

    private fun loadLabels(key: String): List<String> = try {
        assets.open("labels/$key.txt").bufferedReader().useLines { seq ->
            seq.map { it.trim() }.filter { it.isNotEmpty() }.toList()
        }
    } catch (t: Throwable) {
        Log.w(TAG, "Failed to load labels '$key'", t); emptyList()
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraController?.stop()
        if (::engine.isInitialized) engine.release()
    }
}

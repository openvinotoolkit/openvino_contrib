// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package org.intel.openvino.demo.inference

import android.util.Log
import org.intel.openvino.CompiledModel
import org.intel.openvino.Core
import org.intel.openvino.ElementType
import org.intel.openvino.InferRequest
import org.intel.openvino.Layout
import org.intel.openvino.Model
import org.intel.openvino.PrePostProcessor
import org.intel.openvino.ResizeAlgorithm
import org.intel.openvino.Tensor
import org.intel.openvino.demo.postprocess.RawOutput

/**
 * Owns the OpenVINO objects for one model and runs inference on the "CPU" device.
 *
 * The whole preprocessing pipeline is pushed into the compiled model via [PrePostProcessor], using
 * the Java API preprocessing ops added for this demo:
 *  - the app supplies a raw `u8` NHWC RGB frame (built without OpenCV) as a `byte[]` tensor —
 *    `Tensor(ElementType.u8, dims, byte[])`,
 *  - `resize` fits it to the model input (the frame is already letterboxed to square, so this is a
 *    1:1 copy but keeps the graph robust to off-by-one),
 *  - `scale`/`mean` apply the model's normalization (e.g. /255),
 *  - `convert_layout` transposes NHWC → the model's NCHW.
 *
 * Multiple outputs are read back **by index** (Ultralytics exports leave outputs unnamed), using
 * `InferRequest.get_output_tensor(int)`.
 *
 * Not thread-safe: create and call from a single dedicated inference thread (see the analyzer
 * executor in [org.intel.openvino.demo.camera.CameraController]).
 */
class OvEngine(private val core: Core) {

    companion object {
        private const val TAG = "OvEngine"
        const val DEVICE = "CPU"
    }

    private var compiledModel: CompiledModel? = null
    private var inferRequest: InferRequest? = null
    private var config: ModelConfig? = null
    private var numOutputs: Int = 0

    /**
     * Load and compile the model from its downloaded ONNX file ([ModelConfig.onnxPath]). OpenVINO's
     * single-argument read_model reads `.onnx` directly (ONNX frontend built into the runtime).
     * Releases any previously loaded model.
     */
    fun load(config: ModelConfig) {
        release()
        this.config = config

        val model: Model = core.read_model(config.onnxPath)

        // For a dynamic model, pin the spatial input to the session size (derived from the frame
        // aspect) so the graph compiles to a concrete shape. NCHW: [1, 3, H, W].
        if (config.dynamic) {
            model.reshape(intArrayOf(1, 3, config.inputHeight, config.inputWidth))
        }

        // Build the preprocessing graph: user tensor is u8 NHWC of the model's spatial size.
        val ppp = PrePostProcessor(model)
        ppp.input()
            .tensor()
            .set_element_type(ElementType.u8)
            .set_layout(Layout("NHWC"))
            .set_spatial_static_shape(config.inputHeight, config.inputWidth)

        val steps = ppp.input().preprocess()
        steps.resize(ResizeAlgorithm.RESIZE_LINEAR)
        // Scale/mean require a floating-point tensor, but the frame is fed as u8, so convert first.
        steps.convert_element_type(ElementType.f32)
        // Ultralytics: (x - mean) / scale, RGB. mean is 0, scale is 255.
        if (config.mean.any { it != 0f }) {
            steps.mean(config.mean)
        }
        if (config.scale.any { it != 1f }) {
            steps.scale(config.scale)
        }
        steps.convert_layout(Layout(config.layout)) // NHWC -> NCHW
        ppp.input().model().set_layout(Layout(config.layout))
        ppp.build()

        val compiled = core.compile_model(model, DEVICE)
        compiledModel = compiled
        numOutputs = compiled.outputs().size
        inferRequest = compiled.create_infer_request()

        Log.i(
            TAG,
            "Loaded ${config.id} task=${config.task} input=${config.inputWidth}x${config.inputHeight} " +
                "device=$DEVICE outputs=$numOutputs onnx=${config.onnxPath}",
        )
    }

    /**
     * Run inference on a raw `u8` NHWC RGB frame of size [ModelConfig.inputWidth] x
     * [ModelConfig.inputHeight] x 3 and return every output tensor, in index order.
     */
    fun infer(rgbNhwc: ByteArray): List<RawOutput> {
        val cfg = config ?: error("infer() before load()")
        val request = inferRequest ?: error("infer() before load()")

        val dims = intArrayOf(1, cfg.inputHeight, cfg.inputWidth, 3)
        val inputTensor = Tensor(ElementType.u8, dims, rgbNhwc)
        request.set_input_tensor(inputTensor)
        request.infer()

        val results = ArrayList<RawOutput>(numOutputs)
        for (i in 0 until numOutputs) {
            val out = request.get_output_tensor(i)
            results.add(RawOutput(out.data(), out.get_shape()))
        }
        return results
    }

    fun currentConfig(): ModelConfig? = config

    /** Release native resources. Safe to call repeatedly. */
    fun release() {
        try {
            inferRequest?.release()
        } catch (t: Throwable) {
            Log.w(TAG, "Failed to release infer request", t)
        }
        inferRequest = null
        compiledModel = null
        config = null
        numOutputs = 0
    }
}

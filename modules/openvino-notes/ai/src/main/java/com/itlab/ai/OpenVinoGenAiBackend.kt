package com.itlab.ai

import android.annotation.SuppressLint
import android.content.Context
import android.os.SystemClock
import android.util.Log
import com.ovx.openvino.genai.AndroidPipelineProperties
import com.ovx.openvino.genai.ChatHistory
import com.ovx.openvino.genai.DeviceSelection
import com.ovx.openvino.genai.GenerationConfig
import com.ovx.openvino.genai.LLMPipeline
import com.ovx.openvino.genai.StreamingCallback
import com.ovx.openvino.genai.StreamingStatus
import com.ovx.openvino.genai.android.AndroidAssetBundle
import com.ovx.openvino.genai.android.AndroidOpenVinoGenAiRuntime
import java.io.File

@SuppressLint("LogConditional")
class OpenVinoGenAiBackend(
    context: Context,
    private val config: OnDeviceLlmConfig = OnDeviceLlmConfig.defaultAndroid(),
) : LlmInferenceBackend,
    AutoCloseable {
    private val appContext = context.applicationContext
    private var pipeline: LLMPipeline? = null
    private var pipelineCreationCount = 0

    @Synchronized
    override fun generate(
        prompt: String,
        maxNewTokens: Int,
        intent: LlmGenerationIntent,
    ): String {
        val requestStartedAt = SystemClock.elapsedRealtime()
        val coldPipeline = pipeline == null
        val activePipeline = pipeline ?: createPipeline()
        val preparedPrompt = preparePrompt(prompt, config)
        val streamedText = StringBuilder()
        Log.i(
            TAG,
            "generateStart intent=$intent coldPipeline=$coldPipeline " +
                "promptChars=${preparedPrompt.length} maxNewTokens=$maxNewTokens",
        )
        val generationStartedAt = SystemClock.elapsedRealtime()
        val result =
            activePipeline.generate(
                chatHistory(preparedPrompt),
                GenerationConfig
                    .builder()
                    .maxNewTokens(maxNewTokens.toLong())
                    .doSample(false)
                    .applyChatTemplate(true)
                    .build(),
                streamingCallback(intent, streamedText),
            )
        val generationElapsedMs = SystemClock.elapsedRealtime() - generationStartedAt
        logGeneration(
            context =
                GenerationLogContext(
                    intent = intent,
                    coldPipeline = coldPipeline,
                    maxNewTokens = maxNewTokens,
                    promptChars = preparedPrompt.length,
                    requestElapsedMs = SystemClock.elapsedRealtime() - requestStartedAt,
                    generationElapsedMs = generationElapsedMs,
                ),
            result = result,
        )
        val response = result.firstText().orEmpty().ifBlank { streamedText.toString() }
        return if (config.includeReasoningOutput) {
            response
        } else {
            stripReasoningSections(response)
        }
    }

    @Synchronized
    override fun warmUp() {
        if (pipeline != null) {
            return
        }
        val startedAt = SystemClock.elapsedRealtime()
        createPipeline()
        Log.i(TAG, "warmUp elapsedMs=${SystemClock.elapsedRealtime() - startedAt}")
    }

    override fun release() {
        close()
    }

    @Synchronized
    override fun close() {
        pipeline?.close()
        pipeline = null
    }

    @Synchronized
    internal fun diagnostics(): OpenVinoBackendDiagnostics {
        val cacheDir = File(appContext.cacheDir, config.cacheDirName)
        return OpenVinoBackendDiagnostics(
            pipelineCreationCount = pipelineCreationCount,
            cacheDir = cacheDir,
            cacheFileCount = cacheDir.walkTopDown().count { it.isFile },
        )
    }

    private fun createPipeline(): LLMPipeline {
        val startedAt = SystemClock.elapsedRealtime()
        val modelDir = ensureModelDirectory()
        Log.i(TAG, "createPipeline modelDirReady elapsedMs=${SystemClock.elapsedRealtime() - startedAt}")
        val runtimeStartedAt = SystemClock.elapsedRealtime()
        val runtime =
            AndroidOpenVinoGenAiRuntime.prepare(
                appContext,
                AndroidOpenVinoGenAiRuntime
                    .Options
                    .builder()
                    .javaApiLibraryName(config.nativeLibraryName)
                    .build(),
            )
        Log.i(TAG, "createPipeline runtimeReady elapsedMs=${SystemClock.elapsedRealtime() - runtimeStartedAt}")

        val initStartedAt = SystemClock.elapsedRealtime()
        runtime.initialize(config.device)
        Log.i(TAG, "createPipeline javaApiReady elapsedMs=${SystemClock.elapsedRealtime() - initStartedAt}")

        val pipelineStartedAt = SystemClock.elapsedRealtime()
        val cacheDir =
            File(appContext.cacheDir, config.cacheDirName).apply {
                mkdirs()
            }
        val properties =
            AndroidPipelineProperties.cpuLatency(
                cacheDir,
                AndroidPipelineProperties
                    .CpuLatencyOptions
                    .builder()
                    .numStreams(config.numStreams.toLong())
                    .inferenceNumThreads(config.inferenceNumThreads.toLong())
                    .kvCachePrecision(config.kvCachePrecision)
                    .dynamicQuantizationGroupSize(config.dynamicQuantizationGroupSize.toLong())
                    .build(),
            )
        return LLMPipeline(
            modelDir.absolutePath,
            DeviceSelection.of(config.device),
            properties,
        ).also {
            pipeline = it
            pipelineCreationCount += 1
            Log.i(
                TAG,
                "createPipeline pipelineReady elapsedMs=${SystemClock.elapsedRealtime() - pipelineStartedAt} " +
                    "cacheDir=${cacheDir.absolutePath} " +
                    "properties=${properties.toMap()} " +
                    "totalMs=${SystemClock.elapsedRealtime() - startedAt}",
            )
        }
    }

    private fun ensureModelDirectory(): File {
        val targetDir = File(appContext.filesDir, "models/${config.modelDirName}")
        if (!AndroidAssetBundle.directoryExists(appContext, config.assetModelDir)) {
            throw MissingLlmRuntimeException(
                "OpenVINO LLM model assets are missing at assets/${config.assetModelDir}. " +
                    "Gradle should run :ai:stageOpenVinoLlmAssets during preBuild.",
            )
        }

        val assetMarker =
            AndroidAssetBundle.readTextOrNull(
                appContext,
                "${config.assetModelDir}/$MODEL_MARKER_FILE",
            )
        val targetMarker = targetDir.resolve(MODEL_MARKER_FILE).takeIf { it.isFile }?.readText()
        if (targetDir.exists() && !targetDir.list().isNullOrEmpty() && assetMarker == targetMarker) {
            return targetDir
        }

        val startedAt = SystemClock.elapsedRealtime()
        targetDir.deleteRecursively()
        targetDir.mkdirs()
        AndroidAssetBundle.copyDirectory(appContext, config.assetModelDir, targetDir)
        File(appContext.cacheDir, config.cacheDirName).deleteRecursively()
        Log.i(TAG, "copyModelAssets elapsedMs=${SystemClock.elapsedRealtime() - startedAt}")
        return targetDir
    }

    private fun streamingCallback(
        intent: LlmGenerationIntent,
        streamedText: StringBuilder,
    ): StreamingCallback? =
        when (intent) {
            LlmGenerationIntent.Tags,
            ->
                object : StreamingCallback {
                    override fun onText(chunk: String): StreamingStatus {
                        streamedText.append(chunk)
                        return if (shouldStopEarly(intent, streamedText.toString(), config.maxTags)) {
                            StreamingStatus.STOP
                        } else {
                            StreamingStatus.RUNNING
                        }
                    }
                }
            LlmGenerationIntent.Summary,
            LlmGenerationIntent.Rewrite,
            LlmGenerationIntent.General,
            -> null
        }

    private fun chatHistory(prompt: String): ChatHistory =
        ChatHistory
            .builder()
            .addSystem(SYSTEM_PROMPT)
            .addUser(prompt)
            .build()

    private companion object {
        const val TAG = "OpenVinoGenAiBackend"
        const val MODEL_MARKER_FILE = "openvino_llm_manifest.json"
        const val SYSTEM_PROMPT =
            "You are a concise multilingual assistant for a notes app. " +
                "Return only the final user-visible text. " +
                "Never include hidden reasoning, analysis, acknowledgements, or phrases like Okay and Let's."
    }
}

internal data class OpenVinoBackendDiagnostics(
    val pipelineCreationCount: Int,
    val cacheDir: File,
    val cacheFileCount: Int,
)

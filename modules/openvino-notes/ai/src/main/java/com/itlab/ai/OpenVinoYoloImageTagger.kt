package com.itlab.ai

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log
import androidx.core.net.toUri
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.intel.openvino.CompiledModel
import org.intel.openvino.Core
import org.intel.openvino.InferRequest
import org.intel.openvino.Model
import org.intel.openvino.Tensor
import java.io.File
import java.io.IOException

@SuppressLint("LogConditional")
class OpenVinoYoloImageTagger(
    context: Context,
    private val config: OnDeviceVisionConfig = OnDeviceVisionConfig.defaultAndroid(),
) : ImageTaggingBackend,
    AutoCloseable {
    private val appContext = context.applicationContext
    private val assetStore = AndroidAiAssetStore(appContext)
    private val preprocessor = YoloImagePreprocessor(config.inputSize)
    private val parser = YoloOutputParser(config)
    private val deviceProfileProvider = AndroidVisionDeviceProfileProvider(appContext)
    private val lock = Any()

    private var core: Core? = null
    private var model: Model? = null
    private var compiledModel: CompiledModel? = null
    private var inferRequest: InferRequest? = null
    private var classNames: List<String> = emptyList()

    override suspend fun tagImages(imageSources: List<String>): Set<String> =
        withContext(Dispatchers.Default) {
            if (imageSources.isEmpty()) {
                emptySet()
            } else {
                runCatching {
                    synchronized(lock) {
                        ensureInitializedLocked()
                        imageSources
                            .flatMap { source -> detectTagsLocked(source) }
                            .take(config.maxTags)
                            .toSet()
                    }
                }.getOrElse { error ->
                    Log.w(TAG, "OpenVINO YOLO image tagging is unavailable: ${error.message}", error)
                    emptySet()
                }
            }
        }

    private fun ensureInitializedLocked() {
        if (inferRequest != null) return
        System.loadLibrary(config.javaApiLibraryName)

        val modelDir = ensureModelDirectory()
        classNames = loadClassNames(modelDir)
        val selectedModel =
            OnDeviceVisionModelResolver.selectAvailableModel(
                config = config,
                modelRootDir = modelDir,
                profile = deviceProfileProvider.currentProfile(),
            )
        val pluginsFile = ensureRuntimePluginsFile()
        val activeCore =
            if (pluginsFile.isFile) {
                Core(pluginsFile.absolutePath)
            } else {
                Core()
            }
        val modelFile = File(File(modelDir, selectedModel.assetSubdir), selectedModel.modelXmlFileName)
        val activeModel = activeCore.read_model(modelFile.absolutePath) ?: error("Failed to read $modelFile")
        val activeCompiledModel =
            activeCore.compile_model(activeModel, config.device)
                ?: error("Failed to compile ${modelFile.name} for ${config.device}")

        core = activeCore
        model = activeModel
        compiledModel = activeCompiledModel
        inferRequest = activeCompiledModel.create_infer_request()
        Log.i(
            TAG,
            "OpenVINO YOLO image tagger initialized with ${selectedModel.id}: ${modelFile.absolutePath}",
        )
    }

    private fun detectTagsLocked(source: String): Set<String> {
        val bitmap = loadBitmap(source)
        return if (bitmap == null) {
            emptySet()
        } else {
            try {
                val inputData = preprocessor.preprocess(bitmap)
                val inputTensor = Tensor(intArrayOf(1, RGB_CHANNELS, config.inputSize, config.inputSize), inputData)
                val request = inferRequest ?: error("YOLO infer request is not initialized")
                request.set_input_tensor(inputTensor)
                request.infer()
                val outputTensor = request.get_output_tensor()
                if (outputTensor == null) {
                    emptySet()
                } else {
                    parser.parseTags(tensorDataAsFloatArray(outputTensor), classNames)
                }
            } finally {
                bitmap.recycle()
            }
        }
    }

    private fun loadBitmap(source: String): Bitmap? =
        if (source.startsWith(CONTENT_URI_PREFIX)) {
            appContext.contentResolver.openInputStream(source.toUri()).use { input ->
                BitmapFactory.decodeStream(input)
            }
        } else {
            BitmapFactory.decodeFile(source.removePrefix(FILE_URI_PREFIX))
        }

    private fun tensorDataAsFloatArray(tensor: Tensor): FloatArray = tensor.data()

    private fun ensureModelDirectory(): File {
        if (!assetStore.directoryExists(config.assetModelDir)) {
            throw MissingVisionRuntimeException(
                "OpenVINO vision model assets are missing at assets/${config.assetModelDir}. " +
                    "Gradle should run :ai:stageOpenVinoVisionAssets during preBuild.",
            )
        }

        val targetDir = File(appContext.filesDir, "models/${config.modelDirName}")
        val assetMarker = assetStore.readTextOrNull("${config.assetModelDir}/${config.manifestFileName}")
        val targetMarker = targetDir.resolve(config.manifestFileName).takeIf { it.isFile }?.readText()
        if (targetDir.exists() && !targetDir.list().isNullOrEmpty() && assetMarker == targetMarker) {
            return targetDir
        }

        targetDir.deleteRecursively()
        try {
            assetStore.copyDirectory(config.assetModelDir, targetDir)
        } catch (error: IOException) {
            throw MissingVisionRuntimeException("Failed to copy OpenVINO vision model assets.", error)
        }
        return targetDir
    }

    private fun ensureRuntimePluginsFile(): File {
        val targetDir = File(appContext.filesDir, config.runtimeDirName)
        val targetFile = File(targetDir, PLUGINS_FILE)
        if (targetFile.isFile) return targetFile

        return try {
            assetStore.copyFile("${config.runtimeAssetDir}/$PLUGINS_FILE", targetFile)
            targetFile
        } catch (error: IOException) {
            Log.w(TAG, "OpenVINO plugins.xml asset is missing; falling back to default Core().", error)
            File("")
        }
    }

    private fun loadClassNames(modelDir: File): List<String> =
        File(modelDir, config.classNamesFileName)
            .takeIf { it.isFile }
            ?.readLines()
            ?.map { it.trim() }
            ?.filter { it.isNotBlank() }
            .orEmpty()

    override fun release() {
        close()
    }

    override fun close() {
        synchronized(lock) {
            inferRequest?.close()
            compiledModel?.close()
            model?.close()
            core?.close()
            inferRequest = null
            compiledModel = null
            model = null
            core = null
        }
    }

    private companion object {
        const val TAG = "OpenVinoYoloImageTagger"
        const val CONTENT_URI_PREFIX = "content://"
        const val FILE_URI_PREFIX = "file://"
        const val PLUGINS_FILE = "plugins.xml"
        const val RGB_CHANNELS = 3
    }
}

class MissingVisionRuntimeException(
    message: String,
    cause: Throwable? = null,
) : IllegalStateException(message, cause)

package com.itlab.ai

data class OnDeviceVisionConfig(
    val assetModelDir: String,
    val modelDirName: String,
    val classNamesFileName: String,
    val manifestFileName: String,
    val runtimeAssetDir: String,
    val runtimeDirName: String,
    val javaApiLibraryName: String,
    val device: String,
    val preferredModelId: String?,
    val models: List<OnDeviceVisionModelConfig>,
    val inputSize: Int,
    val confidenceThreshold: Float,
    val iouThreshold: Float,
    val maxDetections: Int,
    val maxTags: Int,
) {
    companion object {
        fun defaultAndroid(): OnDeviceVisionConfig =
            OnDeviceVisionConfig(
                assetModelDir = "models/on-device-vision-openvino",
                modelDirName = "on-device-vision-openvino",
                classNamesFileName = "coco.names",
                manifestFileName = "openvino_vision_manifest.json",
                runtimeAssetDir = "openvino-runtime",
                runtimeDirName = "openvino-runtime",
                javaApiLibraryName = "inference_engine_java_api",
                device = "CPU",
                preferredModelId = System.getProperty("openvino.notes.visionModel")?.takeIf { it.isNotBlank() },
                models =
                    listOf(
                        OnDeviceVisionModelConfig(
                            id = "standard",
                            assetSubdir = "standard",
                            modelXmlFileName = "yolo26n.xml",
                            minCpuCores = 4,
                            minRamMb = 2048,
                        ),
                        OnDeviceVisionModelConfig(
                            id = "compact",
                            assetSubdir = "compact",
                            modelXmlFileName = "yolov10n.xml",
                            minCpuCores = 0,
                            minRamMb = 0,
                        ),
                    ),
                inputSize = 640,
                confidenceThreshold = 0.35f,
                iouThreshold = 0.45f,
                maxDetections = 300,
                maxTags = 4,
            )
    }
}

data class OnDeviceVisionModelConfig(
    val id: String,
    val assetSubdir: String,
    val modelXmlFileName: String,
    val minCpuCores: Int,
    val minRamMb: Long,
)

internal data class OnDeviceVisionDeviceProfile(
    val cpuCores: Int,
    val totalRamMb: Long,
)

internal object OnDeviceVisionModelSelector {
    fun select(
        config: OnDeviceVisionConfig,
        profile: OnDeviceVisionDeviceProfile,
    ): OnDeviceVisionModelConfig {
        require(config.models.isNotEmpty()) { "At least one OpenVINO vision model must be configured." }
        config.preferredModelId
            ?.let { preferred -> config.models.firstOrNull { it.id == preferred } }
            ?.let { return it }

        return config.models.firstOrNull { model ->
            profile.cpuCores >= model.minCpuCores &&
                profile.totalRamMb >= model.minRamMb
        } ?: config.models.last()
    }
}

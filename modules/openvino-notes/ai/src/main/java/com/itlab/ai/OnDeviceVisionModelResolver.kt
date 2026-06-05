package com.itlab.ai

import android.app.ActivityManager
import android.content.Context
import java.io.File

internal class AndroidVisionDeviceProfileProvider(
    private val context: Context,
) {
    fun currentProfile(): OnDeviceVisionDeviceProfile {
        val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as? ActivityManager
        val totalRamMb =
            if (activityManager == null) {
                Long.MAX_VALUE
            } else {
                val memoryInfo = ActivityManager.MemoryInfo()
                activityManager.getMemoryInfo(memoryInfo)
                memoryInfo.totalMem / BYTES_PER_MIB
            }

        return OnDeviceVisionDeviceProfile(
            cpuCores = Runtime.getRuntime().availableProcessors(),
            totalRamMb = totalRamMb,
        )
    }

    private companion object {
        const val BYTES_PER_MIB = 1024L * 1024L
    }
}

internal object OnDeviceVisionModelResolver {
    fun selectAvailableModel(
        config: OnDeviceVisionConfig,
        modelRootDir: File,
        profile: OnDeviceVisionDeviceProfile,
    ): OnDeviceVisionModelConfig {
        val availableModels =
            config.models.filter { modelConfig ->
                File(File(modelRootDir, modelConfig.assetSubdir), modelConfig.modelXmlFileName).isFile
            }
        if (availableModels.isEmpty()) {
            throw MissingVisionRuntimeException("No configured OpenVINO vision model is present in $modelRootDir.")
        }

        return OnDeviceVisionModelSelector.select(
            config.copy(models = availableModels),
            profile,
        )
    }
}

package com.itlab.ai

import android.content.Context
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

@RunWith(AndroidJUnit4::class)
class OpenVinoYoloImageTaggerInstrumentedTest {
    @Test(timeout = 240_000)
    fun selectedModelAndCompactFallbackDetectBusOnRealImage() =
        runBlocking {
            val instrumentation = InstrumentationRegistry.getInstrumentation()
            val targetContext = instrumentation.targetContext.applicationContext
            val imagePath = copyTestImage(instrumentation.context, targetContext)
            val config = OnDeviceVisionConfig.defaultAndroid()
            val selectedModel =
                OnDeviceVisionModelSelector.select(
                    config,
                    AndroidVisionDeviceProfileProvider(targetContext).currentProfile(),
                )

            assertBusDetected(targetContext, imagePath, config.copy(preferredModelId = selectedModel.id))

            if (selectedModel.id != COMPACT_MODEL_ID) {
                assertBusDetected(targetContext, imagePath, config.copy(preferredModelId = COMPACT_MODEL_ID))
            }
        }

    private suspend fun assertBusDetected(
        targetContext: Context,
        imagePath: String,
        config: OnDeviceVisionConfig,
    ) {
        val tagger =
            OpenVinoYoloImageTagger(
                targetContext,
                config,
            )
        try {
            val tags = tagger.tagImages(listOf(imagePath))

            assertTrue("${config.preferredModelId} model should detect bus, got $tags", tags.contains("bus"))
        } finally {
            tagger.release()
        }
    }

    private fun copyTestImage(
        testContext: Context,
        targetContext: Context,
    ): String {
        val testImageFile = File(targetContext.filesDir, "bus.jpg")
        if (!testImageFile.exists()) {
            testContext.assets.open("bus.jpg").use { input ->
                testImageFile.outputStream().use { output ->
                    input.copyTo(output)
                }
            }
        }
        return testImageFile.absolutePath
    }

    private companion object {
        const val COMPACT_MODEL_ID = "compact"
    }
}

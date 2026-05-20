package com.itlab.ai

import com.ovx.openvino.genai.AndroidPipelineProperties

data class OnDeviceLlmConfig(
    val modelId: String,
    val assetModelDir: String,
    val modelDirName: String,
    val device: String,
    val nativeLibraryName: String,
    val cacheDirName: String,
    val maxInputChars: Int,
    val summaryMaxInputTokens: Int,
    val summaryMaxNewTokens: Int,
    val tagsMaxInputTokens: Int,
    val tagsMaxNewTokens: Int,
    val rewriteMaxInputTokens: Int,
    val rewriteMaxNewTokens: Int,
    val pipelineMaxPromptTokens: Int,
    val pipelineMinResponseTokens: Int,
    val inferenceNumThreads: Int,
    val numStreams: Int,
    val kvCachePrecision: String,
    val dynamicQuantizationGroupSize: Int,
    val maxTags: Int,
    val approximateCharsPerToken: Int,
    val includeReasoningOutput: Boolean,
    val disableReasoningPromptHint: String,
) {
    companion object {
        fun defaultAndroid(): OnDeviceLlmConfig =
            OnDeviceLlmConfig(
                modelId = "OpenVINO/Qwen3-1.7B-int4-ov",
                assetModelDir = "models/on-device-llm-openvino",
                modelDirName = "on-device-llm-openvino",
                device = "CPU",
                nativeLibraryName = "ov_genai_java_jni",
                cacheDirName = "openvino-genai-cache",
                maxInputChars = 3_000,
                summaryMaxInputTokens = 384,
                summaryMaxNewTokens = 128,
                tagsMaxInputTokens = 320,
                tagsMaxNewTokens = 32,
                rewriteMaxInputTokens = 512,
                rewriteMaxNewTokens = 192,
                pipelineMaxPromptTokens = 640,
                pipelineMinResponseTokens = 64,
                inferenceNumThreads = defaultAndroidInferenceThreads(),
                numStreams = 1,
                kvCachePrecision = "u8",
                dynamicQuantizationGroupSize = 32,
                maxTags = 4,
                approximateCharsPerToken = 4,
                includeReasoningOutput = false,
                disableReasoningPromptHint = "/no_think",
            )

        internal fun defaultAndroidInferenceThreads(
            availableProcessors: Int = Runtime.getRuntime().availableProcessors(),
        ): Int = AndroidPipelineProperties.defaultInferenceThreads(availableProcessors)
    }
}

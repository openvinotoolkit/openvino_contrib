import org.jetbrains.kotlin.gradle.dsl.JvmTarget
import java.net.URI

plugins {
    alias(libs.plugins.android.library)
}

val defaultPythonExecutable =
    providers.provider {
        if (System.getProperty("os.name").contains("Windows", ignoreCase = true)) {
            "python"
        } else {
            "python3"
        }
    }
val pythonExecutable =
    providers
        .gradleProperty("pythonExecutable")
        .orElse(defaultPythonExecutable)

val openvinoAndroidPrebuildRepo =
    providers.gradleProperty("openvinoAndroidPrebuildRepo").orElse("embedded-dev-research/openvino-notes")
val openvinoAndroidPrebuildReleaseTag =
    providers.gradleProperty("openvinoAndroidPrebuildReleaseTag").orElse("openvino-android-prebuilds-nightly")
val openvinoAndroidPrebuildChannel =
    providers.gradleProperty("openvinoAndroidPrebuildChannel").orElse("nightly")
val openvinoAndroidAbi =
    providers.gradleProperty("openvinoAndroidAbi").orElse("arm64-v8a")
val openvinoAndroidCommonArtifactName =
    providers
        .gradleProperty("openvinoAndroidCommonArtifactName")
        .orElse(openvinoAndroidPrebuildChannel.map { "openvino-android-common-$it.zip" })
val openvinoAndroidRuntimeArtifactName =
    providers
        .gradleProperty("openvinoAndroidRuntimeArtifactName")
        .orElse(
            providers.provider {
                "openvino-android-runtime-${openvinoAndroidAbi.get()}-${openvinoAndroidPrebuildChannel.get()}.zip"
            },
        )
val openvinoAndroidCommonPackageName =
    providers
        .gradleProperty("openvinoAndroidCommonPackageName")
        .orElse(openvinoAndroidPrebuildChannel.map { "openvino-android-common-$it" })
val openvinoAndroidRuntimePackageName =
    providers
        .gradleProperty("openvinoAndroidRuntimePackageName")
        .orElse(
            providers.provider {
                "openvino-android-runtime-${openvinoAndroidAbi.get()}-${openvinoAndroidPrebuildChannel.get()}"
            },
        )
val openvinoAndroidPrebuildDownloadDir =
    layout.buildDirectory.dir("openvino/prebuild/download/${openvinoAndroidPrebuildReleaseTag.get()}")
val openvinoAndroidPrebuildExtractDir =
    layout.buildDirectory.dir("openvino/prebuild/extracted/${openvinoAndroidPrebuildReleaseTag.get()}")
val openvinoAndroidCommonArchive =
    openvinoAndroidPrebuildDownloadDir.map { it.file(openvinoAndroidCommonArtifactName.get()) }
val openvinoAndroidCommonArchiveMetadata =
    openvinoAndroidPrebuildDownloadDir.map { it.file("${openvinoAndroidCommonArtifactName.get()}.metadata.json") }
val openvinoAndroidRuntimeArchive =
    openvinoAndroidPrebuildDownloadDir.map { it.file(openvinoAndroidRuntimeArtifactName.get()) }
val openvinoAndroidRuntimeArchiveMetadata =
    openvinoAndroidPrebuildDownloadDir.map { it.file("${openvinoAndroidRuntimeArtifactName.get()}.metadata.json") }
val openvinoAndroidCommonPackageDir =
    openvinoAndroidPrebuildExtractDir.map { it.dir(openvinoAndroidCommonPackageName.get()) }
val openvinoAndroidRuntimePackageDir =
    openvinoAndroidPrebuildExtractDir.map { it.dir(openvinoAndroidRuntimePackageName.get()) }
val openvinoAndroidRuntimeJniRootDir =
    openvinoAndroidRuntimePackageDir.map { it.dir("android-jni") }
val openvinoAndroidRuntimeJniAbiDir =
    openvinoAndroidRuntimePackageDir.map { it.dir("android-jni/${openvinoAndroidAbi.get()}") }
val openvinoRuntimeAssetRootDir = layout.buildDirectory.dir("generated/openvinoRuntimeAssets")
val openvinoRuntimeAssetDir = openvinoRuntimeAssetRootDir.map { it.dir("openvino-runtime") }
val openvinoJavaApiJar =
    openvinoAndroidCommonPackageDir.map {
        it.file("java/openvino-java-api-${openvinoAndroidPrebuildChannel.get()}-android.jar")
    }
val openvinoGenAiJavaApiJar =
    openvinoAndroidCommonPackageDir.map {
        it.file("java/openvino-genai-java-api-main-android.jar")
    }

val onDeviceLlmWeightFormat = providers.gradleProperty("onDeviceLlmWeightFormat").orElse("int4")
val onDeviceLlmBundleRepo =
    providers.gradleProperty("onDeviceLlmBundleRepo").orElse(openvinoAndroidPrebuildRepo)
val onDeviceLlmBundleReleaseTag =
    providers.gradleProperty("onDeviceLlmBundleReleaseTag").orElse("openvino-llm-models-nightly")
val onDeviceLlmBundleArtifactName =
    providers
        .gradleProperty("onDeviceLlmBundleArtifactName")
        .orElse("on-device-llm-openvino-${onDeviceLlmWeightFormat.get()}.zip")
val onDeviceLlmBundleDownloadDir =
    layout.buildDirectory.dir("llm/model-bundle/download/${onDeviceLlmBundleReleaseTag.get()}")
val onDeviceLlmBundleExtractDir =
    layout.buildDirectory.dir("llm/model-bundle/extracted/${onDeviceLlmBundleReleaseTag.get()}")
val onDeviceLlmBundleArchive =
    onDeviceLlmBundleDownloadDir.map { it.file(onDeviceLlmBundleArtifactName.get()) }
val onDeviceLlmBundleArchiveMetadata =
    onDeviceLlmBundleDownloadDir.map { it.file("${onDeviceLlmBundleArtifactName.get()}.metadata.json") }
val onDeviceLlmPreparedDir = providers.gradleProperty("onDeviceLlmPreparedDir")
val resolvedOnDeviceLlmAssetSourceDir =
    onDeviceLlmPreparedDir.orElse(onDeviceLlmBundleExtractDir.map { it.asFile.absolutePath })
val onDeviceLlmAssetRootDir = layout.buildDirectory.dir("generated/openvinoLlmAssets")
val onDeviceLlmAssetDir = onDeviceLlmAssetRootDir.map { it.dir("models/on-device-llm-openvino") }

val onDeviceVisionBundleRepo =
    providers.gradleProperty("onDeviceVisionBundleRepo").orElse(openvinoAndroidPrebuildRepo)
val onDeviceVisionBundleReleaseTag =
    providers.gradleProperty("onDeviceVisionBundleReleaseTag").orElse("openvino-vision-models-nightly")
val onDeviceVisionBundleArtifactName =
    providers
        .gradleProperty("onDeviceVisionBundleArtifactName")
        .orElse("on-device-vision-openvino.zip")
val onDeviceVisionBundleDownloadDir =
    layout.buildDirectory.dir("vision/model-bundle/download/${onDeviceVisionBundleReleaseTag.get()}")
val onDeviceVisionBundleExtractDir =
    layout.buildDirectory.dir("vision/model-bundle/extracted/${onDeviceVisionBundleReleaseTag.get()}")
val onDeviceVisionBundleArchive =
    onDeviceVisionBundleDownloadDir.map { it.file(onDeviceVisionBundleArtifactName.get()) }
val onDeviceVisionBundleArchiveMetadata =
    onDeviceVisionBundleDownloadDir.map { it.file("${onDeviceVisionBundleArtifactName.get()}.metadata.json") }
val onDeviceVisionPreparedDir = providers.gradleProperty("onDeviceVisionPreparedDir")
val resolvedOnDeviceVisionAssetSourceDir =
    onDeviceVisionPreparedDir.orElse(onDeviceVisionBundleExtractDir.map { it.asFile.absolutePath })
val onDeviceVisionAssetRootDir = layout.buildDirectory.dir("generated/openvinoVisionAssets")
val onDeviceVisionAssetDir = onDeviceVisionAssetRootDir.map { it.dir("models/on-device-vision-openvino") }
val visionTestImageAssetRootDir = layout.buildDirectory.dir("generated/openvinoVisionAndroidTestAssets")
val visionTestImageFile = visionTestImageAssetRootDir.map { it.file("bus.jpg") }

android {
    namespace = "com.itlab.ai"
    compileSdk {
        version = release(37)
    }

    defaultConfig {
        minSdk = 33

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        consumerProguardFiles("consumer-rules.pro")

        ndk {
            abiFilters += openvinoAndroidAbi.get()
        }
    }

    sourceSets {
        getByName("main") {
            jniLibs.directories.clear()
            jniLibs.directories.add(openvinoAndroidRuntimeJniRootDir.get().asFile.absolutePath)
            assets.directories.add(openvinoRuntimeAssetRootDir.get().asFile.absolutePath)
            assets.directories.add(onDeviceLlmAssetRootDir.get().asFile.absolutePath)
            assets.directories.add(onDeviceVisionAssetRootDir.get().asFile.absolutePath)
        }
        getByName("androidTest") {
            assets.directories.add(visionTestImageAssetRootDir.get().asFile.absolutePath)
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro",
            )
        }
    }

    packaging {
        jniLibs {
            useLegacyPackaging = true
            pickFirsts += "lib/**/libc++_shared.so"
        }
    }

    lint {
        // The OpenVINO GenAI Android runtime prebuild is selected by openvinoAndroidAbi.
        disable += "ChromeOsAbiSupport"
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}

kotlin {
    compilerOptions {
        jvmTarget.set(JvmTarget.JVM_17)
    }
}

dependencies {
    implementation(
        files(openvinoJavaApiJar, openvinoGenAiJavaApiJar)
            .builtBy("extractOpenVinoAndroidPrebuilds"),
    )
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.appcompat)
    implementation(libs.material)
    implementation(libs.kotlinx.coroutines.core)
    implementation(libs.koin.android)
    implementation(project(":domain"))
    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.espresso.core)
}

val downloadOpenVinoAndroidCommonPrebuild by tasks.registering(Exec::class) {
    group = "ai"
    description = "Download the Android OpenVINO common prebuild from the rolling GitHub prerelease."

    inputs.file(layout.projectDirectory.file("scripts/download_openvino_prebuild.py"))
    outputs.file(openvinoAndroidCommonArchive)
    outputs.file(openvinoAndroidCommonArchiveMetadata)
    outputs.upToDateWhen { false }

    doFirst {
        openvinoAndroidPrebuildDownloadDir.get().asFile.mkdirs()
    }

    commandLine(
        pythonExecutable.get(),
        "scripts/download_openvino_prebuild.py",
        "--repo",
        openvinoAndroidPrebuildRepo.get(),
        "--release-tag",
        openvinoAndroidPrebuildReleaseTag.get(),
        "--artifact-name",
        openvinoAndroidCommonArtifactName.get(),
        "--output",
        openvinoAndroidCommonArchive.get().asFile.absolutePath,
    )
}

val downloadOpenVinoAndroidRuntimePrebuild by tasks.registering(Exec::class) {
    group = "ai"
    description = "Download the Android OpenVINO runtime prebuild from the rolling GitHub prerelease."

    inputs.file(layout.projectDirectory.file("scripts/download_openvino_prebuild.py"))
    outputs.file(openvinoAndroidRuntimeArchive)
    outputs.file(openvinoAndroidRuntimeArchiveMetadata)
    outputs.upToDateWhen { false }

    doFirst {
        openvinoAndroidPrebuildDownloadDir.get().asFile.mkdirs()
    }

    commandLine(
        pythonExecutable.get(),
        "scripts/download_openvino_prebuild.py",
        "--repo",
        openvinoAndroidPrebuildRepo.get(),
        "--release-tag",
        openvinoAndroidPrebuildReleaseTag.get(),
        "--artifact-name",
        openvinoAndroidRuntimeArtifactName.get(),
        "--output",
        openvinoAndroidRuntimeArchive.get().asFile.absolutePath,
    )
}

val extractOpenVinoAndroidPrebuilds by tasks.registering(Copy::class) {
    group = "ai"
    description = "Extract Android OpenVINO prebuilds for Java and native packaging."

    dependsOn(downloadOpenVinoAndroidCommonPrebuild)
    dependsOn(downloadOpenVinoAndroidRuntimePrebuild)
    from({ zipTree(openvinoAndroidCommonArchive.get().asFile) })
    from({ zipTree(openvinoAndroidRuntimeArchive.get().asFile) })
    into(openvinoAndroidPrebuildExtractDir)
    outputs.dir(openvinoAndroidCommonPackageDir)
    outputs.dir(openvinoAndroidRuntimePackageDir)
}

val stageOpenVinoRuntimeAssets by tasks.registering(Copy::class) {
    group = "ai"
    description = "Stage OpenVINO runtime metadata packaged in the Android runtime prebuild."

    dependsOn(extractOpenVinoAndroidPrebuilds)
    from({ openvinoAndroidRuntimeJniAbiDir.get().asFile }) {
        exclude("*.so")
    }
    into(openvinoRuntimeAssetDir)
    outputs.dir(openvinoRuntimeAssetDir)

    doFirst {
        openvinoRuntimeAssetDir.get().asFile.deleteRecursively()
    }
}

val downloadOpenVinoLlmModelBundle by tasks.registering(Exec::class) {
    group = "ai"
    description = "Download the on-device LLM model bundle from the rolling GitHub prerelease."

    onlyIf { !onDeviceLlmPreparedDir.isPresent }
    inputs.file(layout.projectDirectory.file("scripts/download_openvino_prebuild.py"))
    outputs.file(onDeviceLlmBundleArchive)
    outputs.file(onDeviceLlmBundleArchiveMetadata)
    outputs.upToDateWhen { false }

    doFirst {
        onDeviceLlmBundleDownloadDir.get().asFile.mkdirs()
    }

    commandLine(
        pythonExecutable.get(),
        "scripts/download_openvino_prebuild.py",
        "--repo",
        onDeviceLlmBundleRepo.get(),
        "--release-tag",
        onDeviceLlmBundleReleaseTag.get(),
        "--artifact-name",
        onDeviceLlmBundleArtifactName.get(),
        "--output",
        onDeviceLlmBundleArchive.get().asFile.absolutePath,
    )
}

val extractOpenVinoLlmModelBundle by tasks.registering(Copy::class) {
    group = "ai"
    description = "Extract the released on-device LLM model bundle for app assets."

    onlyIf { !onDeviceLlmPreparedDir.isPresent }
    dependsOn(downloadOpenVinoLlmModelBundle)
    from({ zipTree(onDeviceLlmBundleArchive.get().asFile) })
    into(onDeviceLlmBundleExtractDir)
    outputs.dir(onDeviceLlmBundleExtractDir)
}

tasks.register<Copy>("stageOpenVinoLlmAssets") {
    group = "ai"
    description = "Copy the released OpenVINO LLM model into app assets for packaging."
    dependsOn(extractOpenVinoLlmModelBundle)
    onlyIf {
        file(resolvedOnDeviceLlmAssetSourceDir.get()).canonicalFile != onDeviceLlmAssetDir.get().asFile.canonicalFile
    }
    outputs.dir(onDeviceLlmAssetDir)

    from({ file(resolvedOnDeviceLlmAssetSourceDir.get()) })
    into(onDeviceLlmAssetDir)

    doFirst {
        onDeviceLlmAssetDir.get().asFile.deleteRecursively()
    }
}

val downloadOpenVinoVisionModelBundle by tasks.registering(Exec::class) {
    group = "ai"
    description = "Download the on-device OpenVINO vision model bundle from the rolling GitHub prerelease."

    onlyIf { !onDeviceVisionPreparedDir.isPresent }
    inputs.file(layout.projectDirectory.file("scripts/download_openvino_prebuild.py"))
    outputs.file(onDeviceVisionBundleArchive)
    outputs.file(onDeviceVisionBundleArchiveMetadata)
    outputs.upToDateWhen { false }

    doFirst {
        onDeviceVisionBundleDownloadDir.get().asFile.mkdirs()
    }

    commandLine(
        pythonExecutable.get(),
        "scripts/download_openvino_prebuild.py",
        "--repo",
        onDeviceVisionBundleRepo.get(),
        "--release-tag",
        onDeviceVisionBundleReleaseTag.get(),
        "--artifact-name",
        onDeviceVisionBundleArtifactName.get(),
        "--output",
        onDeviceVisionBundleArchive.get().asFile.absolutePath,
    )
}

val extractOpenVinoVisionModelBundle by tasks.registering(Copy::class) {
    group = "ai"
    description = "Extract the released on-device OpenVINO vision model bundle for app assets."

    onlyIf { !onDeviceVisionPreparedDir.isPresent }
    dependsOn(downloadOpenVinoVisionModelBundle)
    from({ zipTree(onDeviceVisionBundleArchive.get().asFile) })
    into(onDeviceVisionBundleExtractDir)
    outputs.dir(onDeviceVisionBundleExtractDir)
}

tasks.register<Copy>("stageOpenVinoVisionAssets") {
    group = "ai"
    description = "Copy the released OpenVINO vision model into app assets for packaging."
    dependsOn(extractOpenVinoVisionModelBundle)
    onlyIf {
        file(resolvedOnDeviceVisionAssetSourceDir.get()).canonicalFile !=
            onDeviceVisionAssetDir.get().asFile.canonicalFile
    }
    outputs.dir(onDeviceVisionAssetDir)

    from({ file(resolvedOnDeviceVisionAssetSourceDir.get()) })
    into(onDeviceVisionAssetDir)

    doFirst {
        onDeviceVisionAssetDir.get().asFile.deleteRecursively()
    }
}

val downloadVisionTestImage by tasks.registering {
    group = "verification"
    description = "Download the Android vision instrumentation test image."

    outputs.file(visionTestImageFile)

    doLast {
        val outputFile = visionTestImageFile.get().asFile
        if (outputFile.isFile && outputFile.length() > 0L) {
            return@doLast
        }

        outputFile.parentFile.mkdirs()
        URI("https://github.com/ultralytics/yolov5/raw/master/data/images/bus.jpg")
            .toURL()
            .openStream()
            .use { input ->
                outputFile.outputStream().use { output ->
                    input.copyTo(output)
                }
            }
    }
}

tasks.named("preBuild") {
    dependsOn(extractOpenVinoAndroidPrebuilds)
    dependsOn(stageOpenVinoRuntimeAssets)
    dependsOn("stageOpenVinoLlmAssets")
    dependsOn("stageOpenVinoVisionAssets")
}

tasks.matching { it.name.startsWith("compile") }.configureEach {
    dependsOn(extractOpenVinoAndroidPrebuilds)
}

tasks.matching { it.name == "preDebugAndroidTestBuild" || it.name == "mergeDebugAndroidTestAssets" }.configureEach {
    dependsOn(downloadVisionTestImage)
}

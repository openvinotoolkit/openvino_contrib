// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

plugins { alias(libs.plugins.android.library) }

android {
    namespace = "com.itlab.ai.text.openvino"
    compileSdk { version = release(37) }
    defaultConfig {
        minSdk = 33
        ndk { abiFilters += providers.gradleProperty("openvinoAndroidAbi").orElse("arm64-v8a").get() }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}

dependencies {
    implementation(project(":ai:text-api"))
    implementation(libs.androidx.core.ktx)
    implementation(libs.kotlinx.coroutines.android)
    testImplementation(libs.junit)
    testImplementation(libs.kotlinx.coroutines.test)
}

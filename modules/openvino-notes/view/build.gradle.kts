// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

plugins {
    alias(libs.plugins.android.library)
    alias(libs.plugins.kotlin.compose)
}

android {
    namespace = "com.openvino.notes.view"
    compileSdk { version = release(37) }
    defaultConfig { minSdk = 33 }
    buildFeatures { compose = true }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}

dependencies {
    implementation(project(":notes:api"))
    implementation(project(":identity:api"))
    implementation(project(":sync:api"))
    implementation(project(":assistant:api"))
    implementation(project(":settings:api"))
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.lifecycle.runtime.compose)
    implementation(libs.androidx.lifecycle.viewmodel.ktx)
    implementation(platform(libs.androidx.compose.bom))
    implementation(libs.androidx.compose.ui)
    implementation(libs.androidx.compose.foundation)
    implementation(libs.androidx.compose.material3)
    implementation(libs.androidx.compose.ui.tooling.preview)
    debugImplementation(libs.androidx.compose.ui.tooling)
    testImplementation(testFixtures(project(":notes:api")))
    testImplementation(testFixtures(project(":identity:api")))
    testImplementation(testFixtures(project(":sync:api")))
    testImplementation(testFixtures(project(":assistant:api")))
    testImplementation(testFixtures(project(":settings:api")))
    testImplementation(libs.junit)
    testImplementation(libs.kotlinx.coroutines.test)
}

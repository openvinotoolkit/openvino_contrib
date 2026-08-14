// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.compose)
}

android {
    namespace = "com.openvino.notes.app"
    compileSdk { version = release(37) }
    defaultConfig {
        applicationId = "com.openvino.notes"
        minSdk = 33
        targetSdk = 37
        versionCode = 1
        versionName = "1.0"
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }
    buildFeatures { compose = true }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    packaging { jniLibs.pickFirsts += "lib/**/libc++_shared.so" }
    testOptions { unitTests.isIncludeAndroidResources = true }
}

dependencies {
    implementation(project(":kernel"))
    implementation(project(":settings:api"))
    implementation(project(":identity:api"))
    implementation(project(":notes:api"))
    implementation(project(":cloud:api"))
    implementation(project(":sync:api"))
    implementation(project(":assistant:api"))
    implementation(project(":ai:text-api"))
    implementation(project(":ai:image-api"))
    implementation(project(":view"))
    implementation(project(":notes:core"))
    implementation(project(":notes:room"))
    implementation(project(":identity:google"))
    implementation(project(":settings:datastore"))
    implementation(project(":cloud:drive"))
    implementation(project(":sync:core"))
    implementation(project(":sync:android"))
    implementation(project(":assistant:core"))
    implementation(project(":ai:text-openvino"))
    implementation(project(":ai:image-openvino"))
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.lifecycle.runtime.ktx)
    implementation(libs.androidx.activity.compose)
    implementation(libs.androidx.work.runtime.ktx)
    implementation(platform(libs.androidx.compose.bom))
    implementation(libs.androidx.compose.ui)
    implementation(libs.androidx.compose.material3)
    implementation(libs.koin.android)
    implementation(libs.koin.androidx.compose)
    debugImplementation(libs.androidx.compose.ui.tooling)
    testImplementation(libs.koin.test)
    testImplementation(libs.androidx.test.core)
    testImplementation(libs.robolectric)
    testImplementation(libs.junit)
}

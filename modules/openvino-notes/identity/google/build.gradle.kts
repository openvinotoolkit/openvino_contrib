// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

plugins { alias(libs.plugins.android.library) }

android {
    namespace = "com.openvino.notes.identity.google"
    compileSdk { version = release(37) }
    defaultConfig { minSdk = 33 }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}

dependencies {
    implementation(project(":identity:api"))
    implementation(libs.androidx.core.ktx)
    implementation(libs.kotlinx.coroutines.android)
    testImplementation(testFixtures(project(":identity:api")))
    testImplementation(project(":kernel"))
    testImplementation(libs.junit)
    testImplementation(libs.kotlinx.coroutines.test)
}

// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

plugins {
    alias(libs.plugins.kotlin.jvm)
    `java-library`
}

dependencies {
    implementation(project(":cloud:api"))
    implementation(project(":identity:api"))
    implementation(project(":kernel"))
    testImplementation(testFixtures(project(":identity:api")))
    testImplementation(libs.junit)
    testImplementation(libs.kotlinx.coroutines.test)
}

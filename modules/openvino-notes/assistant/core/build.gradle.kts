// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

plugins {
    alias(libs.plugins.kotlin.jvm)
    `java-library`
}

dependencies {
    implementation(project(":assistant:api"))
    implementation(project(":notes:api"))
    implementation(project(":ai:text-api"))
    implementation(project(":ai:image-api"))
    testImplementation(testFixtures(project(":notes:api")))
    testImplementation(testFixtures(project(":ai:text-api")))
    testImplementation(testFixtures(project(":ai:image-api")))
    testImplementation(libs.junit)
    testImplementation(libs.kotlinx.coroutines.test)
}

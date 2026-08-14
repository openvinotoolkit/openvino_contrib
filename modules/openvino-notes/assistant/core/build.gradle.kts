// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

plugins {
    alias(libs.plugins.kotlin.jvm)
    `java-library`
}

dependencies {
    api(project(":assistant:api"))
    api(project(":notes:api"))
    api(project(":ai:text-api"))
    api(project(":ai:image-api"))
    api(project(":kernel"))
    testImplementation(testFixtures(project(":notes:api")))
    testImplementation(testFixtures(project(":ai:text-api")))
    testImplementation(testFixtures(project(":ai:image-api")))
    testImplementation(libs.junit)
    testImplementation(libs.kotlinx.coroutines.test)
}


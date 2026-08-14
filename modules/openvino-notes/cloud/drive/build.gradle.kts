// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

plugins {
    alias(libs.plugins.kotlin.jvm)
    `java-library`
}

dependencies {
    api(project(":cloud:api"))
    api(project(":identity:api"))
    api(project(":kernel"))
    testImplementation(testFixtures(project(":identity:api")))
    testImplementation(libs.junit)
    testImplementation(libs.kotlinx.coroutines.test)
}


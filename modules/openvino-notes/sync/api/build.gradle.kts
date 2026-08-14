// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

plugins {
    alias(libs.plugins.kotlin.jvm)
    `java-library`
    `java-test-fixtures`
}

dependencies {
    api(project(":kernel"))
    api(libs.kotlinx.coroutines.core)
    testFixturesImplementation(libs.kotlinx.coroutines.core)
    testImplementation(libs.junit)
}

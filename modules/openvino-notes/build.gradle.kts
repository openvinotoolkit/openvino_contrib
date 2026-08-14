// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import groovy.json.JsonOutput
import org.gradle.api.artifacts.ProjectDependency
import org.gradle.api.plugins.JavaPluginExtension
import org.jetbrains.kotlin.gradle.dsl.JvmTarget
import org.jetbrains.kotlin.gradle.tasks.KotlinJvmCompile

plugins {
    alias(libs.plugins.android.application) apply false
    alias(libs.plugins.android.library) apply false
    alias(libs.plugins.kotlin.jvm) apply false
    alias(libs.plugins.kotlin.compose) apply false
    alias(libs.plugins.ksp) apply false
}

val workspaceBuildRoot = rootDir.resolve("../../../builds/android/openvino-notes/gradle").canonicalFile
layout.buildDirectory.set(workspaceBuildRoot.resolve("root"))

subprojects {
    val relativeProjectPath = path.removePrefix(":").replace(':', '/')
    layout.buildDirectory.set(workspaceBuildRoot.resolve(relativeProjectPath))

    tasks.withType<KotlinJvmCompile>().configureEach {
        compilerOptions.jvmTarget.set(JvmTarget.JVM_17)
    }

    pluginManager.withPlugin("java") {
        extensions.configure<JavaPluginExtension> {
            sourceCompatibility = JavaVersion.VERSION_17
            targetCompatibility = JavaVersion.VERSION_17
        }
    }
}

val expectedModules = setOf(
    ":app", ":view", ":kernel",
    ":settings:api", ":settings:datastore",
    ":identity:api", ":identity:google",
    ":notes:api", ":notes:core", ":notes:room",
    ":cloud:api", ":cloud:drive",
    ":sync:api", ":sync:core", ":sync:android",
    ":assistant:api", ":assistant:core",
    ":ai:text-api", ":ai:text-openvino",
    ":ai:image-api", ":ai:image-openvino",
)

val allowedEdges = setOf(
    ":identity:api->:kernel",
    ":settings:api->:kernel",
    ":notes:api->:kernel", ":notes:api->:identity:api",
    ":cloud:api->:kernel", ":cloud:api->:identity:api",
    ":sync:api->:kernel", ":sync:api->:identity:api",
    ":assistant:api->:kernel", ":assistant:api->:notes:api",
    ":ai:text-api->:kernel", ":ai:image-api->:kernel",
    ":notes:core->:notes:api", ":notes:core->:kernel",
    ":notes:room->:notes:api", ":notes:room->:kernel",
    ":identity:google->:identity:api", ":identity:google->:kernel",
    ":settings:datastore->:settings:api", ":settings:datastore->:kernel",
    ":cloud:drive->:cloud:api", ":cloud:drive->:identity:api", ":cloud:drive->:kernel",
    ":sync:core->:sync:api", ":sync:core->:notes:api", ":sync:core->:cloud:api",
    ":sync:core->:identity:api", ":sync:core->:kernel",
    ":sync:android->:sync:api", ":sync:android->:notes:api", ":sync:android->:kernel",
    ":assistant:core->:assistant:api", ":assistant:core->:notes:api",
    ":assistant:core->:ai:text-api", ":assistant:core->:ai:image-api", ":assistant:core->:kernel",
    ":ai:text-openvino->:ai:text-api", ":ai:text-openvino->:kernel",
    ":ai:image-openvino->:ai:image-api", ":ai:image-openvino->:kernel",
    ":view->:notes:api", ":view->:identity:api", ":view->:sync:api",
    ":view->:assistant:api", ":view->:settings:api", ":view->:kernel",
    ":app->:view", ":app->:notes:core", ":app->:notes:room",
    ":app->:identity:google", ":app->:settings:datastore", ":app->:cloud:drive",
    ":app->:sync:core", ":app->:sync:android", ":app->:assistant:core",
    ":app->:ai:text-openvino", ":app->:ai:image-openvino",
)

fun architectureProjects() = subprojects.filter { it.buildFile.isFile }

fun productionEdges(): Set<String> =
    architectureProjects().flatMap { project ->
        listOf("api", "implementation", "compileOnly").flatMap { configurationName ->
            project.configurations.findByName(configurationName)?.dependencies.orEmpty().mapNotNull { dependency ->
                (dependency as? ProjectDependency)?.path?.let { target -> "${project.path}->$target" }
            }
        }
    }.toSet()

val graphMarkdown = layout.buildDirectory.file("reports/architecture/module-graph.md")
val graphJson = layout.buildDirectory.file("reports/architecture/module-graph.json")

val generateModuleGraph by tasks.registering {
    group = "documentation"
    description = "Generates Mermaid and machine-readable module dependency graphs."
    outputs.files(graphMarkdown, graphJson)

    doLast {
        val edges = productionEdges().sorted()
        val mermaid = buildString {
            appendLine("# OpenVINO Notes Module Graph")
            appendLine()
            appendLine("```mermaid")
            appendLine("graph TD")
            edges.forEach { edge ->
                val (from, to) = edge.split("->")
                val fromId = from.replace(Regex("[^A-Za-z0-9]"), "_")
                val toId = to.replace(Regex("[^A-Za-z0-9]"), "_")
                appendLine("    $fromId[\"$from\"] --> $toId[\"$to\"]")
            }
            appendLine("```")
        }
        val json = mapOf(
            "modules" to architectureProjects().map { it.path }.sorted(),
            "edges" to edges.map { edge ->
                val (from, to) = edge.split("->")
                mapOf("from" to from, "to" to to)
            },
        )
        graphMarkdown.get().asFile.apply { parentFile.mkdirs(); writeText(mermaid) }
        graphJson.get().asFile.apply {
            parentFile.mkdirs()
            writeText(JsonOutput.prettyPrint(JsonOutput.toJson(json)) + "\n")
        }
    }
}

tasks.register("checkArchitecture") {
    group = "verification"
    description = "Validates module dependencies and source ownership boundaries."
    dependsOn(generateModuleGraph)

    doLast {
        val actualModules = architectureProjects().map { it.path }.toSet()
        check(actualModules == expectedModules) {
            "Module set differs. Missing=${expectedModules - actualModules}; unexpected=${actualModules - expectedModules}"
        }

        val actualEdges = productionEdges()
        check(actualEdges == allowedEdges) {
            "Dependency allowlist mismatch. Missing=${allowedEdges - actualEdges}; forbidden=${actualEdges - allowedEdges}"
        }

        val neutralModules = expectedModules.filter {
            it == ":kernel" || it.endsWith(":api") || it.endsWith("-api") ||
                it in setOf(":notes:core", ":sync:core", ":assistant:core", ":cloud:drive")
        }
        neutralModules.forEach { modulePath ->
            project(modulePath).fileTree("src/main").matching { include("**/*.kt") }.files.forEach { source ->
                val text = source.readText()
                check(!Regex("^import androidx?\\.", RegexOption.MULTILINE).containsMatchIn(text)) {
                    "Platform import in $modulePath: ${source.relativeTo(rootDir)}"
                }
            }
        }

        project(":view").fileTree("src/main").matching { include("**/*.kt") }.files.forEach { source ->
            val text = source.readText()
            val forbidden = listOf("androidx.room", "androidx.work", "org.intel.openvino", ".notes.room", ".cloud.drive")
            check(forbidden.none(text::contains)) { "Implementation import in :view: ${source.relativeTo(rootDir)}" }
        }

        architectureProjects().filter { it.path !in setOf(":app", ":view") }.forEach { module ->
            module.fileTree("src/main").matching { include("**/*.kt") }.files.forEach { source ->
                check(!source.readText().contains("org.koin")) {
                    "Koin is only allowed in :app and :view: ${source.relativeTo(rootDir)}"
                }
            }
        }

        val productionAndConfig = files(
            fileTree(rootDir) {
                include("**/src/main/**", "*.gradle.kts", "settings.gradle.kts", "gradle/libs.versions.toml")
                exclude("**/build/**")
            },
        )
        val forbiddenConfigurationTokens = listOf("com.google." + "firebase", "google" + "-services")
        productionAndConfig.files.filter { it.isFile }.forEach { source ->
            val text = source.readText()
            check(forbiddenConfigurationTokens.none(text::contains)) {
                "Firebase or Google Services reference: ${source.relativeTo(rootDir)}"
            }
        }
    }
}

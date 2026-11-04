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
    alias(libs.plugins.ktlint)
    alias(libs.plugins.ksp) apply false
}

val ktlintVersion = libs.versions.ktlint.get()
extensions.configure<org.jlleitschuh.gradle.ktlint.KtlintExtension> {
    version.set(ktlintVersion)
    filter {
        exclude("**/*.gradle.kts")
    }
}
tasks.matching {
    it.name == "ktlintKotlinScriptCheck" || it.name == "ktlintKotlinScriptFormat"
}.configureEach {
    enabled = false
}

val workspaceBuildRoot = (
    providers.gradleProperty("openvinoNotesBuildRoot").orNull?.let(::file)
        ?: rootDir.resolve("build")
    ).canonicalFile
layout.buildDirectory.set(workspaceBuildRoot.resolve("root"))

subprojects {
    apply(plugin = "org.jlleitschuh.gradle.ktlint")
    extensions.configure<org.jlleitschuh.gradle.ktlint.KtlintExtension> {
        version.set(ktlintVersion)
        filter {
            exclude("**/*.gradle.kts")
        }
    }
    tasks.matching {
        it.name == "ktlintKotlinScriptCheck" || it.name == "ktlintKotlinScriptFormat"
    }.configureEach {
        enabled = false
    }

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
    ":notes:api->:kernel",
    ":cloud:api->:kernel",
    ":sync:api->:kernel",
    ":assistant:api->:notes:api",
    ":notes:core->:notes:api", ":notes:core->:kernel",
    ":notes:room->:notes:api", ":notes:room->:kernel",
    ":identity:google->:identity:api", ":identity:google->:kernel",
    ":settings:datastore->:settings:api", ":settings:datastore->:kernel",
    ":cloud:drive->:cloud:api", ":cloud:drive->:identity:api", ":cloud:drive->:kernel",
    ":sync:core->:sync:api", ":sync:core->:notes:api", ":sync:core->:cloud:api",
    ":sync:core->:kernel",
    ":sync:android->:sync:api", ":sync:android->:kernel",
    ":assistant:core->:assistant:api", ":assistant:core->:notes:api",
    ":assistant:core->:ai:text-api", ":assistant:core->:ai:image-api", ":assistant:core->:kernel",
    ":ai:text-openvino->:ai:text-api", ":ai:text-openvino->:kernel",
    ":ai:image-openvino->:ai:image-api", ":ai:image-openvino->:kernel",
    ":view->:notes:api", ":view->:identity:api", ":view->:sync:api",
    ":view->:assistant:api", ":view->:settings:api",
    ":app->:view", ":app->:notes:core", ":app->:notes:room",
    ":app->:identity:google", ":app->:settings:datastore",
    ":app->:sync:core", ":app->:sync:android", ":app->:assistant:core",
    ":app->:ai:text-openvino", ":app->:ai:image-openvino",
    ":app->:kernel", ":app->:settings:api", ":app->:identity:api", ":app->:notes:api",
    ":app->:cloud:api", ":app->:cloud:drive", ":app->:sync:api", ":app->:assistant:api",
    ":app->:ai:text-api", ":app->:ai:image-api",
)

fun architectureProjects() = subprojects.filter { it.buildFile.isFile }

fun isProductionDependencyConfiguration(name: String): Boolean {
    val normalizedName = name.lowercase()
    return !normalizedName.contains("test") &&
        listOf("api", "implementation", "compileonly", "runtimeonly").any(normalizedName::endsWith)
}

fun productionEdges(): Set<String> =
    architectureProjects().flatMap { project ->
        project.configurations.filter { isProductionDependencyConfiguration(it.name) }.flatMap { configuration ->
            configuration.dependencies.mapNotNull { dependency ->
                (dependency as? ProjectDependency)?.path?.let { target -> "${project.path}->$target" }
            }
        }
    }.toSet()

val graphMarkdown = layout.buildDirectory.file("reports/architecture/module-graph.md")
val graphJson = layout.buildDirectory.file("reports/architecture/module-graph.json")
val androidImportPattern = Regex("""^import\s+(?:static\s+)?androidx?\.""", RegexOption.MULTILINE)
val infrastructurePortPattern = Regex("""\bcom\.openvino\.notes\.[A-Za-z0-9_.]+\.api\.port\.""")

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
        val forbiddenEdges = actualEdges - allowedEdges
        check(forbiddenEdges.isEmpty()) {
            "Forbidden module dependencies: ${forbiddenEdges.sorted()}"
        }

        val neutralModules = expectedModules.filter {
            it == ":kernel" || it.endsWith(":api") || it.endsWith("-api") ||
                it in setOf(":notes:core", ":sync:core", ":assistant:core", ":cloud:drive")
        }
        neutralModules.forEach { modulePath ->
            project(modulePath).fileTree("src/main").matching {
                include("**/*.kt", "**/*.java")
            }.files.forEach { source ->
                val text = source.readText()
                check(!androidImportPattern.containsMatchIn(text)) {
                    "Platform import in $modulePath: ${source.relativeTo(rootDir)}"
                }
            }
        }

        project(":view").fileTree("src/main").matching {
            include("**/*.kt", "**/*.java")
        }.files.forEach { source ->
            val text = source.readText()
            val forbidden = listOf("androidx.room", "androidx.work", "org.intel.openvino", ".notes.room", ".cloud.drive")
            check(forbidden.none(text::contains)) { "Implementation import in :view: ${source.relativeTo(rootDir)}" }
            check(!infrastructurePortPattern.containsMatchIn(text)) {
                "Infrastructure API port in :view: ${source.relativeTo(rootDir)}"
            }
        }

        architectureProjects().filter { it.path != ":app" }.forEach { module ->
            module.fileTree("src/main").matching {
                include("**/*.kt", "**/*.java")
            }.files.forEach { source ->
                check(!source.readText().contains("org.koin")) {
                    "Koin is only allowed in :app: ${source.relativeTo(rootDir)}"
                }
            }
            module.configurations.filter { isProductionDependencyConfiguration(it.name) }.forEach { configuration ->
                configuration.dependencies.forEach { dependency ->
                    check(dependency.group != "io.insert-koin") {
                        "Koin dependency is only allowed in :app: ${module.path} -> ${configuration.name} -> ${dependency.group}:${dependency.name}"
                    }
                }
            }
        }

        val productionAndConfig = files(
            fileTree(rootDir) {
                include("**/src/main/**", "**/*.gradle.kts", "settings.gradle.kts", "gradle/libs.versions.toml")
                exclude("**/build/**")
            },
        )
        val scannedConfigurationFiles = productionAndConfig.files
        val unscannedBuildFiles = (listOf(rootProject.buildFile) + architectureProjects().map { it.buildFile })
            .filterNot(scannedConfigurationFiles::contains)
        check(unscannedBuildFiles.isEmpty()) {
            "Gradle build scripts outside Firebase or Google Services scan: " +
                unscannedBuildFiles.joinToString { it.relativeTo(rootDir).path }
        }
        val forbiddenConfigurationTokens = listOf("com.google." + "firebase", "google" + "-services")
        scannedConfigurationFiles.filter { it.isFile }.forEach { source ->
            val text = source.readText()
            check(forbiddenConfigurationTokens.none(text::contains)) {
                "Firebase or Google Services reference: ${source.relativeTo(rootDir)}"
            }
        }

        val requiredPackagePrefix = "com.openvino.notes"
        fileTree(rootDir) {
            include("**/src/**/*.kt")
            exclude("**/build/**")
        }.files.forEach { source ->
            val packageName = Regex("^package\\s+([^\\s]+)", RegexOption.MULTILINE)
                .find(source.readText())
                ?.groupValues
                ?.get(1)
            check(packageName == requiredPackagePrefix || packageName?.startsWith("$requiredPackagePrefix.") == true) {
                "Unexpected package in ${source.relativeTo(rootDir)}: $packageName"
            }
        }
    }
}

tasks.register("checkArchitectureRules") {
    group = "verification"
    description = "Self-tests subset semantics used by the architecture dependency gate."
    doLast {
        listOf("api", "implementation", "compileOnly", "runtimeOnly", "debugImplementation", "freeReleaseRuntimeOnly")
            .forEach { configurationName ->
                check(isProductionDependencyConfiguration(configurationName)) {
                    "Production dependency configuration must be checked: $configurationName"
                }
            }
        listOf("testImplementation", "androidTestRuntimeOnly", "debugUnitTestImplementation", "testFixturesApi")
            .forEach { configurationName ->
                check(!isProductionDependencyConfiguration(configurationName)) {
                    "Test dependency configuration must not enter the production graph: $configurationName"
                }
            }
        val permittedSubset = allowedEdges.take(1).toSet()
        check((permittedSubset - allowedEdges).isEmpty()) { "An allowed optional edge subset must pass" }
        val futureCloudComposition = setOf(":app->:cloud:api", ":app->:cloud:drive")
        check((futureCloudComposition - allowedEdges).isEmpty()) { "Future Cloud composition edges must remain allowed" }
        check(":sync:core->:identity:api" !in allowedEdges) { "Sync core must use Kernel account scope and Cloud contracts" }
        val syntheticForbidden = setOf(":view->:notes:room")
        check((syntheticForbidden - allowedEdges) == syntheticForbidden) { "A forbidden edge must be rejected" }
        check(infrastructurePortPattern.containsMatchIn("import com.openvino.notes.notes.api.port.NotesRepository")) {
            "Infrastructure port imports must be recognized"
        }
        check(infrastructurePortPattern.containsMatchIn("com.openvino.notes.sync.api.port.SyncCheckpointPort")) {
            "Fully qualified infrastructure port references must be recognized"
        }
        check(infrastructurePortPattern.containsMatchIn("import com.openvino.notes.notes.api.port.RemoteNoteChange")) {
            "Infrastructure supporting models must be recognized"
        }
        check(!infrastructurePortPattern.containsMatchIn("import com.openvino.notes.notes.api.NotesService")) {
            "Consumer service imports must remain allowed"
        }
        check(androidImportPattern.containsMatchIn("import android.content.Context")) {
            "Android imports must be recognized"
        }
        check(androidImportPattern.containsMatchIn("import static android.os.Build.VERSION.SDK_INT;")) {
            "Java static Android imports must be recognized"
        }
        check(!androidImportPattern.containsMatchIn("import com.android.test.FakeAndroid")) {
            "Non-Android imports must remain allowed"
        }
    }
}

tasks.named("checkArchitecture") { dependsOn("checkArchitectureRules") }

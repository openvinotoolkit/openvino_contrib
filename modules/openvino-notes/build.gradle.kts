import com.android.build.api.dsl.ApplicationExtension
import com.android.build.api.dsl.CommonExtension
import com.android.build.api.dsl.LibraryExtension
import io.gitlab.arturbosch.detekt.Detekt
import io.gitlab.arturbosch.detekt.extensions.DetektExtension
import kotlinx.kover.gradle.plugin.dsl.KoverProjectExtension
import org.jlleitschuh.gradle.ktlint.KtlintExtension

// Top-level build file where you can add configuration options common to all sub-projects/modules.
plugins {
    alias(libs.plugins.android.application) apply false
    alias(libs.plugins.kotlin.compose) apply false
    alias(libs.plugins.android.library) apply false
    alias(libs.plugins.kotlin.serialization) apply false
    alias(libs.plugins.ktlint)
    alias(libs.plugins.detekt)
    alias(libs.plugins.kover)
    id("com.google.gms.google-services") version "4.4.4" apply false
}

configure<KtlintExtension> {
    android.set(true)
    ignoreFailures.set(false)
    outputToConsole.set(true)
}

configure<DetektExtension> {
    buildUponDefaultConfig = true
    config.setFrom(files("$rootDir/detekt.yml"))
    basePath = rootDir.absolutePath
}

tasks.withType<Detekt>().configureEach {
    reports {
        html.required.set(true)
        md.required.set(false)
        sarif.required.set(true)
        txt.required.set(false)
        xml.required.set(true)
    }
}

kover {
    reports {
        filters {
            excludes {
                classes(
                    "*.BuildConfig",
                    "*.Manifest*",
                    "*.R",
                    "*.R$*",
                    "com.itlab.notes.App*",
                    "com.itlab.notes.MainActivity*",
                    "com.itlab.notes.di.*",
                    "com.itlab.notes.ui.editor.*",
                    "com.itlab.notes.ui.notes.*",
                    "com.itlab.notes.ui.theme.*",
                )
            }
        }
        verify {
            rule {
                minBound(60)
            }
        }
    }
}

subprojects {
    apply(plugin = "org.jlleitschuh.gradle.ktlint")
    apply(plugin = "io.gitlab.arturbosch.detekt")
    apply(plugin = "org.jetbrains.kotlinx.kover")

    dependencyLocking {
        lockAllConfigurations()
    }

    configure<KtlintExtension> {
        android.set(true)
        ignoreFailures.set(false)
        outputToConsole.set(true)
    }

    configure<DetektExtension> {
        buildUponDefaultConfig = true
        config.setFrom(rootProject.files("detekt.yml"))
        basePath = rootDir.absolutePath
    }

    configure<KoverProjectExtension> {
        reports {
            filters {
                excludes {
                    classes(
                        "*.BuildConfig",
                        "*.Manifest*",
                        "*.R",
                        "*.R$*",
                        "com.itlab.notes.App*",
                        "com.itlab.notes.MainActivity*",
                        "com.itlab.notes.di.*",
                        "com.itlab.notes.ui.editor.*",
                        "com.itlab.notes.ui.notes.*",
                        "com.itlab.notes.ui.theme.*",
                    )
                }
            }
            verify {
                rule {
                    minBound(60)
                }
            }
        }
    }

    tasks.withType<Detekt>().configureEach {
        jvmTarget = "21"
        reports {
            html.required.set(true)
            md.required.set(false)
            sarif.required.set(true)
            txt.required.set(false)
            xml.required.set(true)
        }
    }

    pluginManager.withPlugin("com.android.application") {
        extensions.configure<ApplicationExtension> {
            configureAndroidLint()
        }
    }

    pluginManager.withPlugin("com.android.library") {
        extensions.configure<LibraryExtension> {
            configureAndroidLint()
        }
    }

    if (name == "app") {
        tasks.matching { it.name == "koverVerify" }.configureEach {
            enabled = false
        }
    }
}

fun CommonExtension.configureAndroidLint() {
    lint.apply {
        abortOnError = true
        checkAllWarnings = true
        checkDependencies = true
        checkTestSources = true
        htmlReport = true
        lintConfig = rootProject.file("lint.xml")
        warningsAsErrors = true
        xmlReport = true
    }
}

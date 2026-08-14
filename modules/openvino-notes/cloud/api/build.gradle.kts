plugins {
    alias(libs.plugins.kotlin.jvm)
    `java-library`
    `java-test-fixtures`
}

dependencies {
    api(project(":kernel"))
    api(project(":identity:api"))
    api(libs.kotlinx.coroutines.core)
    testFixturesImplementation(libs.kotlinx.coroutines.core)
    testImplementation(libs.junit)
}


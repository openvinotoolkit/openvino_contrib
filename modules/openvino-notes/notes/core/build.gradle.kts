plugins {
    alias(libs.plugins.kotlin.jvm)
    `java-library`
}

dependencies {
    api(project(":notes:api"))
    api(project(":kernel"))
    testImplementation(testFixtures(project(":notes:api")))
    testImplementation(testFixtures(project(":identity:api")))
    testImplementation(libs.junit)
    testImplementation(libs.kotlinx.coroutines.test)
}


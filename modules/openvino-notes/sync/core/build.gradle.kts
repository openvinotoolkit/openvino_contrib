plugins {
    alias(libs.plugins.kotlin.jvm)
    `java-library`
}

dependencies {
    api(project(":sync:api"))
    api(project(":notes:api"))
    api(project(":cloud:api"))
    api(project(":identity:api"))
    api(project(":kernel"))
    testImplementation(testFixtures(project(":sync:api")))
    testImplementation(testFixtures(project(":notes:api")))
    testImplementation(testFixtures(project(":cloud:api")))
    testImplementation(testFixtures(project(":identity:api")))
    testImplementation(libs.junit)
    testImplementation(libs.kotlinx.coroutines.test)
}


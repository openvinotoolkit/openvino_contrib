pluginManagement {
    repositories {
        google {
            content {
                includeGroupByRegex("com\\.android.*")
                includeGroupByRegex("com\\.google.*")
                includeGroupByRegex("androidx.*")
            }
        }
        mavenCentral()
        gradlePluginPortal()
    }
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
    }
}

rootProject.name = "openvino-notes"

include(":app")
include(":view")
include(":kernel")
include(":settings:api")
include(":settings:datastore")
include(":identity:api")
include(":identity:google")
include(":notes:api")
include(":notes:core")
include(":notes:room")
include(":cloud:api")
include(":cloud:drive")
include(":sync:api")
include(":sync:core")
include(":sync:android")
include(":assistant:api")
include(":assistant:core")
include(":ai:text-api")
include(":ai:text-openvino")
include(":ai:image-api")
include(":ai:image-openvino")


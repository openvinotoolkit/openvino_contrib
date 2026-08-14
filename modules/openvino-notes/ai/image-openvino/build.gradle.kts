plugins { alias(libs.plugins.android.library) }

android {
    namespace = "com.itlab.ai.image.openvino"
    compileSdk { version = release(37) }
    defaultConfig {
        minSdk = 33
        ndk { abiFilters += providers.gradleProperty("openvinoAndroidAbi").orElse("arm64-v8a").get() }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}

dependencies {
    api(project(":ai:image-api"))
    api(project(":kernel"))
    implementation(libs.androidx.core.ktx)
    implementation(libs.kotlinx.coroutines.android)
    testImplementation(libs.junit)
}


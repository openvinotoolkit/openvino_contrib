# On-Device AI

`openvino-notes` runs text assistance on Android through OpenVINO GenAI. The app uses the `:ai` module for model packaging, runtime bootstrap, prompt construction, output cleanup, and validation.

## Scope

Implemented text actions:

- summary suggestion
- tag suggestion
- note rewrite
- model warm-up and release when the editor opens or closes

Out of scope for this backend:

- image understanding and image tagging
- server-side inference
- committing model weights or OpenVINO runtime binaries to git

## Runtime Packaging

The Android app consumes external release assets at build time:

- OpenVINO Android common prebuild: `openvino-android-common-nightly.zip`
- OpenVINO Android ABI runtime prebuild: `openvino-android-runtime-<abi>-nightly.zip`
- OpenVINO LLM model bundle: `openvino-llm-models-nightly`

Gradle downloads and extracts these assets during `:ai:preBuild`:

```bash
./gradlew :app:assembleRelease
```

Useful overrides:

```bash
./gradlew :app:assembleRelease \
  -PopenvinoAndroidAbi=x86_64 \
  -PonDeviceLlmPreparedDir=/path/to/extracted/openvino-llm-model
```

The app build does not build OpenVINO, OpenVINO GenAI, the GenAI Java JNI bridge, or model IR files locally. Runtime and Java API binaries come from the OpenVINO Android prebuild release assets; the model comes from the LLM model bundle release asset.

The release APK packages:

- OpenVINO, OpenVINO GenAI, and GenAI Java JNI native libraries under `lib/<abi>`
- OpenVINO Java API and GenAI Java API jars from the common prebuild
- runtime metadata under `assets/openvino-runtime`
- model files under `assets/models/on-device-llm-openvino`

The default model bundle is `OpenVINO/Qwen3-1.7B-int4-ov`.

## Source Boundaries

`genai-java-api` owns generic wrapper/runtime behavior:

- JNI bridge CMake target
- Android native-library preload and plugin registration helpers
- Android runtime asset staging helpers
- `PipelineProperties`, Android CPU defaults, and performance metric wrappers

`openvino-notes` owns app-specific behavior:

- note prompts and language hints
- text extraction from notes
- result normalization, retry, and fallback policy
- model bundle choice and release-asset wiring
- UI state for warm-up, generation progress, and accepted suggestions

Keep generic OpenVINO GenAI wrapper code out of this repository unless it is only an application integration point.

## Validation

Host checks:

```bash
./gradlew :ai:testDebugUnitTest :domain:testDebugUnitTest --stacktrace
```

Release packaging check:

```bash
./gradlew :app:assembleRelease --stacktrace
```

On-device model quality check:

```bash
./gradlew :ai:connectedDebugAndroidTest \
  -Pandroid.testInstrumentationRunnerArguments.class=com.itlab.ai.OnDeviceLlmMultilingualInstrumentedTest \
  --stacktrace
```

The default connected test packages the `arm64-v8a` OpenVINO runtime. Use `-PopenvinoAndroidAbi=x86_64` when validating on an x86_64 Android target. The Gradle host can be macOS arm64, Linux x86_64, or another CI machine as long as it can build the APK and reach a compatible Android device over ADB.

The test validates complex Russian, English, German, and French notes and checks that warm generations stay within the configured performance envelope.

## Privacy And Publication Rules

- Do not commit signing keys, keystore properties, downloaded models, extracted runtime bundles, APKs, or device logs.
- Keep local paths in `local.properties` or Gradle properties, not in committed source.
- Do not log note text or generated model output from production code.
- If a validation needs real device output, keep only aggregate timings and pass/fail status in public artifacts.

# Local CI Reproduction

This guide explains how to reproduce repository checks locally on Linux, macOS, and Windows.

## Toolchain Baseline

- JDK 17
- Android SDK command-line tools
- Android platform-tools
- Android platform `android-36`
- Android build-tools `36.0.0`
- Git

## Validation Status

### macOS arm64

Validated locally:

- foundation checks
- debug build and unit tests
- coverage
- release assemble
- release lint
- CodeQL build
- preflight
- APK task validation
- gitleaks
- emulator instrumentation

Environment used:

- Homebrew `openjdk@17`
- Android SDK root: `~/Library/Android/sdk`
- emulator target: `android-34`, `google_apis`, `arm64-v8a`

### Linux arm64

Validated locally on Ubuntu 24.04 arm64:

- JDK 17 setup
- Android SDK command-line tools setup
- `sdkmanager` package install
- preflight

Observed differences:

- if `local.properties` points to an SDK path from another OS, Gradle prefers it over `ANDROID_SDK_ROOT`
- `.github/scripts/security/run_gitleaks.sh` fails on arm64 because it downloads a `linux_x64` binary
- Android lint/build path failed with `Aapt2InternalException: Failed to start AAPT2 process`

### Linux x86_64

This is the closest path to GitHub CI and should be treated as the reference Linux environment for full parity.

## Environment Setup

### macOS

```bash
export JAVA_HOME="$(brew --prefix openjdk@17)/libexec/openjdk.jdk/Contents/Home"
export PATH="$JAVA_HOME/bin:$PATH"
export ANDROID_SDK_ROOT="$HOME/Library/Android/sdk"
export ANDROID_HOME="$ANDROID_SDK_ROOT"
export ANDROID_API_LEVEL=36
export ANDROID_BUILD_TOOLS=36.0.0
export INSTALL_SYSTEM_IMAGE=false
export ANDROID_SYSTEM_IMAGE=

bash .github/scripts/setup/install_android_sdk_packages.sh
```

### Linux

```bash
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
export PATH="$JAVA_HOME/bin:$PATH"
export ANDROID_SDK_ROOT="$HOME/Android/Sdk"
export ANDROID_HOME="$ANDROID_SDK_ROOT"
export ANDROID_API_LEVEL=36
export ANDROID_BUILD_TOOLS=36.0.0
export INSTALL_SYSTEM_IMAGE=false
export ANDROID_SYSTEM_IMAGE=

bash .github/scripts/setup/install_android_sdk_packages.sh
```

If your Linux host is `arm64`, native SDK installation can still work, but Android Gradle tasks may differ from `x86_64` CI behavior.

### Windows PowerShell

```powershell
$env:ANDROID_API_LEVEL = "36"
$env:ANDROID_BUILD_TOOLS = "36.0.0"
$env:INSTALL_SYSTEM_IMAGE = "false"
$env:ANDROID_SYSTEM_IMAGE = ""

.\.github\scripts\setup\install_android_sdk_packages_windows.ps1
```

## Cross-OS SDK Path Note

If the working tree was copied from another machine or another OS, check `local.properties`.

If it contains a stale `sdk.dir`, Gradle uses that value before `ANDROID_SDK_ROOT`.

Fix it with:

```bash
printf 'sdk.dir=%s\n' "$HOME/Android/Sdk" > local.properties
```

On macOS:

```bash
printf 'sdk.dir=%s\n' "$HOME/Library/Android/sdk" > local.properties
```

## Main Local Gate

Use this as the default pre-push validation path.

### Linux or macOS

```bash
bash .github/scripts/quality/run_foundation.sh
bash .github/scripts/quality/run_debug_build_and_unit_tests.sh
bash .github/scripts/quality/run_coverage.sh
```

### Windows PowerShell

```powershell
.\.github\scripts\quality\run_debug_build_and_unit_tests_windows.ps1

.\gradlew.bat ktlintCheck detekt ai:lintDebug app:lintDebug data:lintDebug domain:lintDebug ai:testDebugUnitTest app:testDebugUnitTest data:testDebugUnitTest domain:testDebugUnitTest koverXmlReport koverVerify --stacktrace
```

## Checks

### Foundation

CI script: [run_foundation.sh](/Users/anesterov/repos/openvino-notes/.github/scripts/quality/run_foundation.sh)

Tasks:

- `ktlintCheck`
- `detekt`
- `ai:lintDebug`
- `app:lintDebug`
- `data:lintDebug`
- `domain:lintDebug`

Linux or macOS:

```bash
bash .github/scripts/quality/run_foundation.sh
```

Windows:

```powershell
.\gradlew.bat ktlintCheck detekt ai:lintDebug app:lintDebug data:lintDebug domain:lintDebug --stacktrace
```

Outputs:

- `**/build/reports/detekt/`
- `**/build/reports/ktlint/`
- `**/build/reports/lint-results-*.html`
- `**/build/reports/lint-results-*.xml`

Linux arm64 note:

- on Ubuntu 24.04 arm64, the Android lint/build path failed with `Aapt2InternalException: Failed to start AAPT2 process`

### Debug Build and Host Unit Tests

CI scripts:

- [run_debug_build_and_unit_tests.sh](/Users/anesterov/repos/openvino-notes/.github/scripts/quality/run_debug_build_and_unit_tests.sh)
- [run_debug_build_and_unit_tests_windows.ps1](/Users/anesterov/repos/openvino-notes/.github/scripts/quality/run_debug_build_and_unit_tests_windows.ps1)

Tasks:

- `ai:assembleDebug`
- `app:assembleDebug`
- `app:assembleDebugAndroidTest`
- `data:assembleDebug`
- `domain:assembleDebug`
- `ai:testDebugUnitTest`
- `app:testDebugUnitTest`
- `data:testDebugUnitTest`
- `domain:testDebugUnitTest`

Linux or macOS:

```bash
bash .github/scripts/quality/run_debug_build_and_unit_tests.sh
```

Windows:

```powershell
.\.github\scripts\quality\run_debug_build_and_unit_tests_windows.ps1
```

Outputs:

- `app/build/outputs/apk/debug/app-debug.apk`
- `app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk`
- `**/build/test-results/`
- `**/build/reports/tests/`
- `**/build/outputs/unit_test_code_coverage/`

### Coverage

CI script: [run_coverage.sh](/Users/anesterov/repos/openvino-notes/.github/scripts/quality/run_coverage.sh)

Tasks:

- `ai:testDebugUnitTest`
- `app:testDebugUnitTest`
- `data:testDebugUnitTest`
- `domain:testDebugUnitTest`
- `koverXmlReport`
- `koverVerify`

Linux or macOS:

```bash
bash .github/scripts/quality/run_coverage.sh
```

Windows:

```powershell
.\gradlew.bat ai:testDebugUnitTest app:testDebugUnitTest data:testDebugUnitTest domain:testDebugUnitTest koverXmlReport koverVerify --stacktrace
```

Output:

- `**/build/reports/kover/`

### Release

CI scripts:

- [assemble_release.sh](/Users/anesterov/repos/openvino-notes/.github/scripts/release/assemble_release.sh)
- [lint_release.sh](/Users/anesterov/repos/openvino-notes/.github/scripts/release/lint_release.sh)

Linux or macOS:

```bash
bash .github/scripts/release/assemble_release.sh
bash .github/scripts/release/lint_release.sh
```

Windows:

```powershell
.\gradlew.bat ai:assembleRelease app:assembleRelease data:assembleRelease domain:assembleRelease --stacktrace
.\gradlew.bat ai:lintRelease app:lintRelease data:lintRelease domain:lintRelease --stacktrace
```

### Secrets

CI script: [run_gitleaks.sh](/Users/anesterov/repos/openvino-notes/.github/scripts/security/run_gitleaks.sh)

Linux x86_64:

```bash
bash .github/scripts/security/run_gitleaks.sh
```

Native `gitleaks` path for Linux arm64, macOS, and Windows:

```bash
gitleaks detect --source . --report-format sarif --report-path build/reports/gitleaks/gitleaks.sarif --redact
```

Linux arm64 note:

- `.github/scripts/security/run_gitleaks.sh` failed with `cannot execute binary file: Exec format error`

### Preflight

CI script: [classify_changes.sh](/Users/anesterov/repos/openvino-notes/.github/scripts/preflight/classify_changes.sh)

The script expects `GITHUB_OUTPUT`, so a plain local invocation is not enough.

Example:

```bash
tmpfile="$(mktemp)"
GITHUB_OUTPUT="$tmpfile" \
EVENT_NAME=pull_request \
BASE_SHA="$(git merge-base upstream/main HEAD)" \
HEAD_SHA="$(git rev-parse HEAD)" \
BEFORE_SHA="" \
CURRENT_SHA="$(git rev-parse HEAD)" \
bash .github/scripts/preflight/classify_changes.sh
cat "$tmpfile"
rm -f "$tmpfile"
```

### CodeQL Build

CI script: [build_for_codeql.sh](/Users/anesterov/repos/openvino-notes/.github/scripts/codeql/build_for_codeql.sh)

Linux or macOS:

```bash
bash .github/scripts/codeql/build_for_codeql.sh
```

Windows:

```powershell
.\gradlew.bat clean ai:assembleDebug app:assembleDebug data:assembleDebug domain:assembleDebug --no-build-cache --rerun-tasks --stacktrace
```

Limitation:

- this reproduces the build input
- full local parity still requires local CodeQL CLI setup

### Android Instrumentation

CI scripts:

- [validate_debug_apk_tasks.sh](/Users/anesterov/repos/openvino-notes/.github/scripts/android/validate_debug_apk_tasks.sh)
- [run_emulator_instrumentation.sh](/Users/anesterov/repos/openvino-notes/.github/scripts/android/run_emulator_instrumentation.sh)
- [run_all_instrumentation_variants.sh](/Users/anesterov/repos/openvino-notes/.github/scripts/android/run_all_instrumentation_variants.sh)

GitHub CI target:

- Linux
- API 34
- `x86_64`
- `pixel_7`

Validated local macOS target:

- macOS arm64
- API 34
- `arm64-v8a`
- `pixel_7`

Linux arm64 status:

- not validated end-to-end
- upstream Android build path already failed at AAPT2 startup

Task validation:

Linux or macOS:

```bash
bash .github/scripts/android/validate_debug_apk_tasks.sh
```

Windows:

```powershell
.\gradlew.bat app:assembleDebug app:assembleDebugAndroidTest -m --stacktrace
```

Basic manual flow:

```bash
./gradlew app:assembleDebug app:assembleDebugAndroidTest --stacktrace
APK_DIR=app/build/outputs/apk bash .github/scripts/android/run_emulator_instrumentation.sh
```

Validated macOS arm64 flow:

```bash
export JAVA_HOME="$(brew --prefix openjdk@17)/libexec/openjdk.jdk/Contents/Home"
export PATH="$JAVA_HOME/bin:$PATH:$HOME/Library/Android/sdk/emulator:$HOME/Library/Android/sdk/platform-tools"
export ANDROID_SDK_ROOT="$HOME/Library/Android/sdk"
export ANDROID_HOME="$ANDROID_SDK_ROOT"

printf 'no\n' | avdmanager create avd -n pixel7api34Arm64Local -k 'system-images;android-34;google_apis;arm64-v8a' -d 'pixel_7'
./gradlew app:assembleDebug app:assembleDebugAndroidTest --stacktrace
emulator -avd pixel7api34Arm64Local -no-snapshot-save -no-window -noaudio -no-boot-anim -gpu swiftshader_indirect &
adb wait-for-device
until [ "$(adb shell getprop sys.boot_completed 2>/dev/null | tr -d '\r')" = "1" ]; do sleep 5; done
ANDROID_SERIAL=emulator-5554 APK_DIR=app/build/outputs/apk bash .github/scripts/android/run_emulator_instrumentation.sh
adb -s emulator-5554 emu kill
```

Validated result:

```text
com.itlab.notes.ExampleInstrumentedTest:.

Time: 0.006

OK (1 test)
```

## Not Fully Reproducible Locally

- dependency review
- workflow summaries
- artifact upload
- full CodeQL action lifecycle

## Practical Rule

For pre-push debugging, reproducing the Gradle tasks and repository scripts above is usually enough.

On Windows, use Git Bash or WSL for the bash-based Android helper scripts. The Gradle commands themselves are platform-specific, but the repository's emulator helper scripts are shell scripts, not PowerShell scripts.

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$openvinoAndroidAbi = if ($env:OPENVINO_ANDROID_ABI) { $env:OPENVINO_ANDROID_ABI } else { "arm64-v8a" }

& .\gradlew.bat `
  "-PopenvinoAndroidAbi=$openvinoAndroidAbi" `
  ai:assembleDebug `
  app:assembleDebug `
  app:assembleDebugAndroidTest `
  data:assembleDebug `
  domain:assembleDebug `
  ai:testDebugUnitTest `
  app:testDebugUnitTest `
  data:testDebugUnitTest `
  domain:testDebugUnitTest `
  --stacktrace

if ($LASTEXITCODE -ne 0) {
    throw "Gradle debug build and unit tests failed with exit code $LASTEXITCODE."
}

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

& .\gradlew.bat `
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

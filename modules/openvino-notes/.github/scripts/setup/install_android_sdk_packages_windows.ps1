Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-SdkManagerPath {
    $candidates = @(
        (Join-Path $env:ANDROID_SDK_ROOT "cmdline-tools\latest\bin\sdkmanager.bat"),
        (Join-Path $env:ANDROID_HOME "cmdline-tools\latest\bin\sdkmanager.bat")
    )

    foreach ($candidate in $candidates) {
        if ($candidate -and (Test-Path $candidate)) {
            return $candidate
        }
    }

    $command = Get-Command sdkmanager.bat -ErrorAction SilentlyContinue
    if ($command) {
        return $command.Source
    }

    throw "sdkmanager.bat was not found in ANDROID_SDK_ROOT, ANDROID_HOME, or PATH."
}

$sdkmanager = Get-SdkManagerPath

$platformPackage = "platforms;android-$env:ANDROID_API_LEVEL"
$sdkPackages = & $sdkmanager --list
if ($sdkPackages -notmatch [regex]::Escape($platformPackage)) {
    $fallbackPlatformPackage = "$platformPackage.0"
    if ($sdkPackages -match [regex]::Escape($fallbackPlatformPackage)) {
        $platformPackage = $fallbackPlatformPackage
    }
}

$packages = @(
    $platformPackage,
    "build-tools;$env:ANDROID_BUILD_TOOLS"
)

if ($env:INSTALL_SYSTEM_IMAGE -eq "true" -and -not [string]::IsNullOrWhiteSpace($env:ANDROID_SYSTEM_IMAGE)) {
    $packages += "emulator"
    $packages += $env:ANDROID_SYSTEM_IMAGE
}

$licenseAnswers = 1..20 | ForEach-Object { "y" }
$licenseAnswers | & $sdkmanager --licenses *> $null

& $sdkmanager @packages

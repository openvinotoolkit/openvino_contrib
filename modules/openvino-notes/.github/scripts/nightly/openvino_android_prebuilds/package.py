from __future__ import annotations

import hashlib
import json
import os
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from .common import BuildConfig, run, write_env_file


def zip_directory(source_dir: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(source_dir.rglob("*")):
            archive.write(path, path.relative_to(source_dir.parent))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_key_value_manifest(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def write_common_source_manifest(source: Path, target: Path) -> None:
    android_keys = ("android_abi=", "android_platform=", "android_ndk_version=")
    lines = [line for line in source.read_text(encoding="utf-8").splitlines() if not line.startswith(android_keys)]
    target.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def copy_unique_file(source: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / source.name
    if target.exists():
        if sha256_file(source) == sha256_file(target):
            return
        raise SystemExit(f"Refusing to overwrite different library with the same name: {target}")
    shutil.copy2(source, target)


def strip_shared_libraries(config: BuildConfig, root: Path) -> None:
    if not config.llvm_strip.is_file():
        raise SystemExit(f"llvm-strip is missing: {config.llvm_strip}")

    for library in sorted(root.rglob("*.so")):
        run([str(config.llvm_strip), "--strip-unneeded", str(library)])


def copy_runtime_payload(config: BuildConfig, runtime_dir: Path, jni_dir: Path) -> None:
    ndk_libcxx = config.llvm_prebuilt_dir / "sysroot" / "usr" / "lib" / config.android_target_triple / "libc++_shared.so"
    if not ndk_libcxx.is_file():
        raise SystemExit(f"Android NDK libc++ runtime not found: {ndk_libcxx}")

    copy_unique_file(ndk_libcxx, jni_dir)

    runtime_lib_dir = runtime_dir / "lib"
    runtime_libraries = sorted(runtime_lib_dir.rglob("*.so"))
    if not runtime_libraries:
        raise SystemExit(f"No OpenVINO runtime shared libraries found under {runtime_lib_dir}")
    for library in runtime_libraries:
        copy_unique_file(library, jni_dir)

    tbb_lib_dir = runtime_dir / "3rdparty" / "tbb" / "lib"
    for library in sorted(tbb_lib_dir.glob("*.so")):
        copy_unique_file(library, jni_dir)

    plugin_xml_files = sorted(runtime_lib_dir.rglob("plugins.xml"))
    if not plugin_xml_files:
        raise SystemExit(f"No OpenVINO plugins.xml files found under {runtime_lib_dir}")
    for plugins_xml in plugin_xml_files:
        if plugins_xml.parent.name.startswith("openvino-"):
            plugins_dir = jni_dir / plugins_xml.parent.name
        elif plugins_xml.parent.parent == runtime_lib_dir:
            plugins_dir = jni_dir
        else:
            plugins_dir = jni_dir / plugins_xml.relative_to(runtime_lib_dir).parent
        plugins_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(plugins_xml, plugins_dir)


def copy_genai_java_api_payload(config: BuildConfig, jni_dir: Path) -> None:
    jni_library = config.genai_java_api_install_dir / "lib" / "libov_genai_java_jni.so"
    if not jni_library.is_file():
        raise SystemExit(f"OpenVINO GenAI Java API JNI library is missing: {jni_library}")
    copy_unique_file(jni_library, jni_dir)


def package_asset(path: Path, *, package_type: str, abi: str | None) -> dict[str, object]:
    return {
        "name": path.name,
        "type": package_type,
        "abi": abi,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def write_readme(path: Path, text: str) -> None:
    path.write_text(text.strip() + "\n", encoding="utf-8")


def package_common(config: BuildConfig, source_manifest: Path) -> dict[str, object]:
    java_dir = config.common_package_root / "java"
    metadata_dir = config.common_package_root / "metadata"
    java_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(config.artifacts_dir / "java-api" / f"openvino-java-api-{config.package_ref}-android.jar", java_dir)
    shutil.copy2(
        config.artifacts_dir / "genai-java-api" / f"openvino-genai-java-api-{config.genai_java_api_package_ref}-android.jar",
        java_dir,
    )
    write_common_source_manifest(source_manifest, metadata_dir / "source-manifest.txt")
    write_readme(
        config.common_package_root / "README.md",
        f"""
# OpenVINO Android common prebuild

Contents:
- `java/`: Java API classes jar for `org.intel.openvino.*`.
- `java/`: OpenVINO GenAI Java API classes jar for `com.ovx.openvino.genai.*`.
- `metadata/source-manifest.txt`: exact source refs and commits used for this build.

Use this package together with one `openvino-android-runtime-<abi>-{config.package_ref}.zip` package.
""",
    )

    zip_directory(config.common_package_root, config.common_zip_path)
    return package_asset(config.common_zip_path, package_type="common", abi=None)


def package_runtime(config: BuildConfig, runtime_dir: Path, source_manifest: Path) -> dict[str, object]:
    jni_dir = config.runtime_package_root / "android-jni" / config.android_abi
    metadata_dir = config.runtime_package_root / "metadata"
    jni_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    copy_runtime_payload(config, runtime_dir, jni_dir)
    copy_genai_java_api_payload(config, jni_dir)
    strip_shared_libraries(config, config.runtime_package_root)
    shutil.copy2(source_manifest, metadata_dir / "source-manifest.txt")
    write_readme(
        config.runtime_package_root / "README.md",
        f"""
# OpenVINO Android {config.android_abi} runtime prebuild

Contents:
- `android-jni/{config.android_abi}/`: shared libraries ready to copy into an Android app `src/main/jniLibs/{config.android_abi}` directory, including `libc++_shared.so` from the Android NDK.
- `android-jni/{config.android_abi}/libov_genai_java_jni.so`: OpenVINO GenAI Java API JNI bridge built in `REAL` mode.
- `android-jni/{config.android_abi}/plugins.xml`: OpenVINO plugin registry for arch-directory runtime installs.
- `android-jni/{config.android_abi}/openvino-*/plugins.xml`: OpenVINO plugin registry files for versioned runtime installs.
- `metadata/source-manifest.txt`: exact source refs and commits used for this build.

This package is built for Android {config.android_abi}, Android platform {config.android_platform}, and Android NDK {config.android_ndk_version}.
""",
    )

    zip_directory(config.runtime_package_root, config.runtime_zip_path)
    return package_asset(config.runtime_zip_path, package_type="runtime", abi=config.android_abi)


def package_prebuild(config: BuildConfig) -> None:
    config.export_runtime_environment()
    runtime_dir = config.install_dir / "openvino-android" / "runtime"
    source_manifest = config.artifacts_dir / "source-manifest.txt"
    if not runtime_dir.is_dir():
        raise SystemExit(f"OpenVINO install runtime directory is missing: {runtime_dir}")
    if not source_manifest.is_file():
        raise SystemExit(f"Source manifest is missing: {source_manifest}")

    package_parent = config.artifacts_dir / "package"
    if package_parent.exists():
        shutil.rmtree(package_parent)

    assets: list[dict[str, object]] = []
    if config.package_common:
        assets.append(package_common(config, source_manifest))
    assets.append(package_runtime(config, runtime_dir, source_manifest))

    package_manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": read_key_value_manifest(source_manifest),
        "android": {
            "abi": config.android_abi,
            "platform": config.android_platform,
            "ndk_version": config.android_ndk_version,
        },
        "assets": assets,
    }
    config.package_manifest_path.write_text(json.dumps(package_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    for asset in assets:
        print(f"{asset['name']} {asset['size_bytes']} bytes sha256={asset['sha256']}")
    print(f"{config.package_manifest_path} {config.package_manifest_path.stat().st_size} bytes")

    write_env_file(
        os.environ.get("GITHUB_OUTPUT"),
        {
            "artifact_path": str(config.artifacts_dir),
            "artifact_name": f"openvino-android-{config.android_abi}-prebuilds",
            "package_name": config.runtime_package_name,
        },
    )


def ccache_stats(config: BuildConfig) -> None:
    if shutil.which("ccache") is None:
        return

    config.artifacts_dir.mkdir(parents=True, exist_ok=True)
    run(["ccache", "--show-stats"], log=config.artifacts_dir / "ccache-stats.log")

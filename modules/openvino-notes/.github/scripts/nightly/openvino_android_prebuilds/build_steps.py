from __future__ import annotations

import shutil

from .common import BuildConfig, cmake_android_args, command_output, require_command, run


def build_onetbb(config: BuildConfig) -> None:
    config.export_runtime_environment()
    require_command("cmake")
    require_command("ccache")
    run(
        [
            "cmake",
            "-S",
            str(config.src_dir / "oneTBB"),
            "-B",
            str(config.build_dir / "oneTBB"),
            "-G",
            "Ninja",
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_INSTALL_PREFIX={config.install_dir / 'oneTBB'}",
            *cmake_android_args(config),
            "-DTBB_TEST=OFF",
            "-DTBB_EXAMPLES=OFF",
            "-DCMAKE_SHARED_LINKER_FLAGS=-Wl,--undefined-version",
            "-DCMAKE_C_COMPILER_LAUNCHER=ccache",
            "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
        ],
        log=config.artifacts_dir / "onetbb-configure.log",
    )
    run(["cmake", "--build", str(config.build_dir / "oneTBB")], log=config.artifacts_dir / "onetbb-build.log")
    run(["cmake", "--install", str(config.build_dir / "oneTBB")], log=config.artifacts_dir / "onetbb-install.log")


def configure_openvino(config: BuildConfig) -> None:
    config.export_runtime_environment()
    require_command("cmake")
    require_command("ccache")
    run(
        [
            "cmake",
            "-S",
            str(config.src_dir / "openvino"),
            "-B",
            str(config.build_dir / "openvino-android"),
            "-G",
            "Ninja",
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_INSTALL_PREFIX={config.install_dir / 'openvino-android'}",
            *cmake_android_args(config),
            f"-DOPENVINO_EXTRA_MODULES={config.src_dir / 'openvino_contrib' / 'modules' / 'java_api'};{config.src_dir / 'openvino.genai'}",
            "-DTHREADING=TBB",
            f"-DTBB_DIR={config.install_dir / 'oneTBB' / 'lib' / 'cmake' / 'TBB'}",
            "-DENABLE_INTEL_GPU=OFF",
            "-DENABLE_INTEL_NPU=OFF",
            "-DENABLE_TEMPLATE=OFF",
            "-DENABLE_TESTS=OFF",
            "-DENABLE_FUNCTIONAL_TESTS=OFF",
            "-DENABLE_SAMPLES=OFF",
            "-DENABLE_TOOLS=OFF",
            "-DENABLE_PYTHON=OFF",
            "-DENABLE_OV_ONNX_FRONTEND=OFF",
            "-DENABLE_OV_PADDLE_FRONTEND=OFF",
            "-DENABLE_OV_PYTORCH_FRONTEND=OFF",
            "-DENABLE_OV_TF_FRONTEND=OFF",
            "-DENABLE_OV_TF_LITE_FRONTEND=OFF",
            "-DENABLE_OV_JAX_FRONTEND=OFF",
            "-DENABLE_OV_IR_FRONTEND=ON",
            "-DENABLE_PLUGINS_XML=ON",
            "-DENABLE_SNIPPETS_LIBXSMM_TPP=OFF",
            "-DENABLE_GGUF=ON",
            "-DENABLE_CLANG_FORMAT=OFF",
            "-DENABLE_CLANG_TIDY=OFF",
            "-DCMAKE_C_COMPILER_LAUNCHER=ccache",
            "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
        ],
        log=config.artifacts_dir / "openvino-configure.log",
    )


def build_openvino_runtime(config: BuildConfig) -> None:
    config.export_runtime_environment()
    require_command("cmake")
    run(
        [
            "cmake",
            "--build",
            str(config.build_dir / "openvino-android"),
            "--target",
            "openvino",
            "openvino_c",
            "ov_frontends",
            "ov_plugins",
        ],
        log=config.artifacts_dir / "openvino-runtime-build.log",
    )
    validate_acl_archive(config)


def build_openvino_genai(config: BuildConfig) -> None:
    config.export_runtime_environment()
    require_command("cmake")
    run(
        [
            "cmake",
            "--build",
            str(config.build_dir / "openvino-android"),
            "--target",
            "openvino_tokenizers",
            "openvino_genai",
            "openvino_genai_c",
        ],
        log=config.artifacts_dir / "openvino-genai-build.log",
    )


def build_openvino_java_api(config: BuildConfig) -> None:
    config.export_runtime_environment()
    require_command("cmake")
    require_command("javac")
    require_command("jar")
    run(
        [
            "cmake",
            "--build",
            str(config.build_dir / "openvino-android"),
            "--target",
            "inference_engine_java_api",
        ],
        log=config.artifacts_dir / "openvino-java-jni-build.log",
    )
    build_java_api_jar(config)


def validate_acl_archive(config: BuildConfig) -> None:
    if config.android_abi != "arm64-v8a":
        message = f"ACL archive check skipped for non-ARM ABI: {config.android_abi}"
        print(message)
        (config.artifacts_dir / "acl-archive-check.log").write_text(message + "\n", encoding="utf-8")
        return

    acl_archive = (
        config.build_dir
        / "openvino-android"
        / "src"
        / "plugins"
        / "intel_cpu"
        / "thirdparty"
        / "acl_build"
        / "build"
        / "arm64-v8.2-a"
        / "libarm_compute-static.a"
    )
    if not acl_archive.is_file() or acl_archive.stat().st_size == 0:
        raise SystemExit(f"ACL archive is missing or empty: {acl_archive}")

    object_count = len(command_output([str(config.llvm_prebuilt_dir / "bin" / "llvm-ar"), "t", str(acl_archive)]).splitlines())
    if object_count <= 0:
        raise SystemExit(f"ACL archive contains no objects: {acl_archive}")

    message = f"ACL archive object count: {object_count}"
    print(message)
    (config.artifacts_dir / "acl-archive-check.log").write_text(message + "\n", encoding="utf-8")


def build_java_api_jar(config: BuildConfig) -> None:
    java_out = config.artifacts_dir / "java-api"
    jar_path = java_out / f"openvino-java-api-{config.package_ref}-android.jar"
    if java_out.exists():
        shutil.rmtree(java_out)
    (java_out / "classes").mkdir(parents=True)

    sources = sorted((config.src_dir / "openvino_contrib" / "modules" / "java_api" / "src" / "main" / "java").rglob("*.java"))
    source_list = java_out / "java-sources.txt"
    source_list.write_text("\n".join(str(source) for source in sources) + "\n", encoding="utf-8")
    run(["javac", "--release", "11", "-d", str(java_out / "classes"), f"@{source_list}"])
    run(["jar", "--create", "--file", str(jar_path), "-C", str(java_out / "classes"), "."])


def install_openvino(config: BuildConfig) -> None:
    config.export_runtime_environment()
    require_command("cmake")
    run(["cmake", "--install", str(config.build_dir / "openvino-android")], log=config.artifacts_dir / "openvino-install.log")


def build_genai_java_api(config: BuildConfig) -> None:
    config.export_runtime_environment()
    require_command("cmake")
    require_command("ccache")
    require_command("javac")
    require_command("jar")
    if not config.openvino_runtime_cmake_dir.is_dir():
        raise SystemExit(f"OpenVINO runtime CMake package directory is missing: {config.openvino_runtime_cmake_dir}")

    run(
        [
            "cmake",
            "-S",
            str(config.src_dir / "genai-java-api"),
            "-B",
            str(config.genai_java_api_build_dir),
            "-G",
            "Ninja",
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_INSTALL_PREFIX={config.genai_java_api_install_dir}",
            *cmake_android_args(config),
            "-DOV_GENAI_JNI_MODE=REAL",
            f"-DOpenVINO_DIR={config.openvino_runtime_cmake_dir}",
            f"-DOpenVINOGenAI_DIR={config.openvino_runtime_cmake_dir}",
            f"-DCMAKE_PREFIX_PATH={config.openvino_runtime_cmake_dir}",
            "-DCMAKE_C_COMPILER_LAUNCHER=ccache",
            "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
        ],
        log=config.artifacts_dir / "genai-java-api-configure.log",
    )
    run(
        ["cmake", "--build", str(config.genai_java_api_build_dir), "--target", "ov_genai_java_jni"],
        log=config.artifacts_dir / "genai-java-api-native-build.log",
    )
    run(["cmake", "--install", str(config.genai_java_api_build_dir)], log=config.artifacts_dir / "genai-java-api-install.log")
    build_genai_java_api_jar(config)


def build_genai_java_api_jar(config: BuildConfig) -> None:
    if not config.android_platform_jar.is_file():
        raise SystemExit(f"Android platform jar is missing: {config.android_platform_jar}")

    source_root = config.src_dir / "genai-java-api"
    java_out = config.artifacts_dir / "genai-java-api"
    jar_path = java_out / f"openvino-genai-java-api-{config.genai_java_api_package_ref}-android.jar"
    if java_out.exists():
        shutil.rmtree(java_out)
    (java_out / "classes").mkdir(parents=True)

    source_dirs = [source_root / "src" / "main" / "java", source_root / "src" / "android" / "java"]
    sources = sorted(source for source_dir in source_dirs for source in source_dir.rglob("*.java"))
    if not sources:
        raise SystemExit(f"No GenAI Java API sources found under {source_root}")
    source_list = java_out / "java-sources.txt"
    source_list.write_text("\n".join(str(source) for source in sources) + "\n", encoding="utf-8")
    run(
        [
            "javac",
            "--release",
            "17",
            "-classpath",
            str(config.android_platform_jar),
            "-d",
            str(java_out / "classes"),
            f"@{source_list}",
        ]
    )
    run(["jar", "--create", "--file", str(jar_path), "-C", str(java_out / "classes"), "."])

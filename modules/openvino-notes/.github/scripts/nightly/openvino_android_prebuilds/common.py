from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from subprocess import PIPE, STDOUT

from plumbum import local
from plumbum.commands import CommandNotFound, ProcessExecutionError
from pydantic import AliasChoices, Field, ValidationError, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


SCRIPT_ROOT = Path(__file__).resolve().parents[4]
SUPPORTED_ANDROID_ABIS = frozenset({"arm64-v8a", "x86_64"})
ANDROID_ABI_TARGET_TRIPLES = {
    "arm64-v8a": "aarch64-linux-android",
    "x86_64": "x86_64-linux-android",
}


def default_run_root() -> Path:
    runner_temp = os.environ.get("RUNNER_TEMP", "")
    if runner_temp:
        return Path(runner_temp) / "openvino-android-prebuilds"
    return SCRIPT_ROOT / ".tmp" / "openvino-android-prebuilds"


def require_command(name: str) -> None:
    try:
        local.which(name)
    except CommandNotFound as error:
        raise SystemExit(f"Required command is missing: {name}") from error


def _command(args: list[str]):
    if not args:
        raise SystemExit("Command argument list must not be empty.")

    try:
        return local[args[0]][args[1:]]
    except CommandNotFound as error:
        raise SystemExit(f"Required command is missing: {args[0]}") from error


def run(args: list[str], *, log: Path | None = None, cwd: Path | None = None) -> None:
    print("+ " + " ".join(args), flush=True)
    command = _command(args)
    log_file = None

    try:
        if log is not None:
            log.parent.mkdir(parents=True, exist_ok=True)
            log_file = log.open("w", encoding="utf-8")

        process = command.popen(cwd=str(cwd) if cwd else None, stdout=PIPE, stderr=STDOUT, text=True, bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            if log_file is not None:
                log_file.write(line)
        return_code = process.wait()
    finally:
        if log_file is not None:
            log_file.close()

    if return_code != 0:
        raise SystemExit(f"Command failed with exit code {return_code}: {' '.join(args)}")


def command_output(args: list[str]) -> str:
    try:
        _, stdout, _ = _command(args).run()
    except ProcessExecutionError as error:
        raise SystemExit(f"Command failed with exit code {error.retcode}: {' '.join(args)}") from error
    return stdout.strip()


def write_env_file(path: str | None, entries: dict[str, str]) -> None:
    if not path:
        return

    with Path(path).open("a", encoding="utf-8") as env_file:
        for key, value in entries.items():
            env_file.write(f"{key}={value}\n")


def append_path_file(path: str | None, value: Path) -> None:
    if not path:
        return

    with Path(path).open("a", encoding="utf-8") as path_file:
        path_file.write(f"{value}\n")


def safe_filename_part(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip(".-")
    return safe or "unknown"


class BuildConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", extra="ignore", populate_by_name=True)

    openvino_ref: str = Field("android-mbind-compat", validation_alias="OPENVINO_REF")
    openvino_genai_ref: str = Field("master", validation_alias="OPENVINO_GENAI_REF")
    genai_java_api_ref: str = Field("main", validation_alias="GENAI_JAVA_API_REF")
    openvino_contrib_ref: str = Field("master", validation_alias="OPENVINO_CONTRIB_REF")
    onetbb_ref: str = Field("v2023.0.0", validation_alias="ONETBB_REF")
    package_channel: str = Field("nightly", validation_alias="PACKAGE_CHANNEL")
    openvino_repo: str = Field("https://github.com/embedded-dev-research/openvino.git", validation_alias="OPENVINO_REPO")
    openvino_contrib_repo: str = Field(
        "https://github.com/openvinotoolkit/openvino_contrib.git",
        validation_alias="OPENVINO_CONTRIB_REPO",
    )
    openvino_genai_repo: str = Field(
        "https://github.com/openvinotoolkit/openvino.genai.git",
        validation_alias="OPENVINO_GENAI_REPO",
    )
    genai_java_api_repo: str = Field(
        "https://github.com/ov-ext-labs/genai-java-api.git",
        validation_alias="GENAI_JAVA_API_REPO",
    )
    onetbb_repo: str = Field("https://github.com/uxlfoundation/oneTBB.git", validation_alias="ONETBB_REPO")
    android_abi: str = Field("arm64-v8a", validation_alias="ANDROID_ABI")
    android_platform: str = Field("33", validation_alias="ANDROID_PLATFORM")
    android_ndk_version: str = Field("29.0.14206865", validation_alias="ANDROID_NDK_VERSION")
    package_common: bool = Field(True, validation_alias=AliasChoices("PACKAGE_COMMON", "PUBLISH_COMMON_PACKAGE"))
    android_sdk_root: Path = Field(validation_alias=AliasChoices("ANDROID_SDK_ROOT", "ANDROID_HOME"))
    android_ndk: Path | None = Field(default=None, validation_alias="ANDROID_NDK")
    run_root: Path = Field(default_factory=default_run_root, validation_alias="RUN_ROOT")
    ccache_dir: Path | None = Field(default=None, validation_alias="CCACHE_DIR")

    @model_validator(mode="after")
    def validate_paths(self) -> BuildConfig:
        if self.android_abi not in SUPPORTED_ANDROID_ABIS:
            supported = ", ".join(sorted(SUPPORTED_ANDROID_ABIS))
            raise ValueError(f"Unsupported Android ABI '{self.android_abi}'. Supported 64-bit ABIs: {supported}")

        versioned_ndk = self.android_sdk_root / "ndk" / self.android_ndk_version
        self.android_ndk = versioned_ndk if versioned_ndk.is_dir() else self.android_ndk or versioned_ndk
        if not self.android_ndk.is_dir():
            raise ValueError(f"Android NDK not found: {self.android_ndk}")

        if str(self.run_root) in {"", "/"}:
            raise ValueError(f"RUN_ROOT must point to a disposable build directory, got: '{self.run_root}'")

        if self.ccache_dir is None:
            runner_temp = os.environ.get("RUNNER_TEMP", "")
            self.ccache_dir = (Path(runner_temp) if runner_temp else self.run_root) / "ccache"

        return self

    @classmethod
    def from_env(cls) -> BuildConfig:
        try:
            return cls()
        except ValidationError as error:
            raise SystemExit(f"Invalid build environment:\n{error}") from error

    @property
    def src_dir(self) -> Path:
        return self.run_root / "src"

    @property
    def build_dir(self) -> Path:
        return self.run_root / "build"

    @property
    def install_dir(self) -> Path:
        return self.run_root / "install"

    @property
    def artifacts_dir(self) -> Path:
        return self.run_root / "artifacts"

    @property
    def package_name(self) -> str:
        return self.runtime_package_name

    @property
    def package_ref(self) -> str:
        return safe_filename_part(self.package_channel)

    @property
    def genai_java_api_package_ref(self) -> str:
        return safe_filename_part(self.genai_java_api_ref)

    @property
    def common_package_name(self) -> str:
        return f"openvino-android-common-{self.package_ref}"

    @property
    def runtime_package_name(self) -> str:
        return f"openvino-android-runtime-{self.android_abi}-{self.package_ref}"

    @property
    def package_root(self) -> Path:
        return self.runtime_package_root

    @property
    def common_package_root(self) -> Path:
        return self.artifacts_dir / "package" / self.common_package_name

    @property
    def runtime_package_root(self) -> Path:
        return self.artifacts_dir / "package" / self.runtime_package_name

    @property
    def zip_path(self) -> Path:
        return self.runtime_zip_path

    @property
    def common_zip_path(self) -> Path:
        return self.artifacts_dir / f"{self.common_package_name}.zip"

    @property
    def runtime_zip_path(self) -> Path:
        return self.artifacts_dir / f"{self.runtime_package_name}.zip"

    @property
    def genai_java_api_build_dir(self) -> Path:
        return self.build_dir / "genai-java-api"

    @property
    def genai_java_api_install_dir(self) -> Path:
        return self.install_dir / "genai-java-api"

    @property
    def openvino_runtime_cmake_dir(self) -> Path:
        return self.install_dir / "openvino-android" / "runtime" / "cmake"

    @property
    def android_platform_jar(self) -> Path:
        return self.android_sdk_root / "platforms" / f"android-{self.android_platform}" / "android.jar"

    @property
    def package_manifest_path(self) -> Path:
        return self.artifacts_dir / f"openvino-android-package-manifest-{self.android_abi}.json"

    @property
    def android_target_triple(self) -> str:
        return ANDROID_ABI_TARGET_TRIPLES[self.android_abi]

    @property
    def llvm_prebuilt_dir(self) -> Path:
        prebuilt_root = self.android_ndk / "toolchains" / "llvm" / "prebuilt"
        host_glob = "darwin-*" if sys.platform == "darwin" else "linux-*"
        matches = sorted(prebuilt_root.glob(host_glob))
        if not matches:
            raise SystemExit(f"Could not locate Android NDK LLVM prebuilt tools under {self.android_ndk}")
        return matches[0]

    @property
    def llvm_strip(self) -> Path:
        return self.llvm_prebuilt_dir / "bin" / "llvm-strip"

    def export_runtime_environment(self) -> None:
        os.environ["ANDROID_NDK"] = str(self.android_ndk)
        os.environ["CCACHE_DIR"] = str(self.ccache_dir)
        os.environ["PATH"] = f"{self.llvm_prebuilt_dir / 'bin'}{os.pathsep}{os.environ['PATH']}"


def cmake_android_args(config: BuildConfig) -> list[str]:
    return [
        f"-DCMAKE_TOOLCHAIN_FILE={config.android_ndk / 'build' / 'cmake' / 'android.toolchain.cmake'}",
        f"-DANDROID_ABI={config.android_abi}",
        f"-DANDROID_PLATFORM={config.android_platform}",
        "-DANDROID_STL=c++_shared",
    ]

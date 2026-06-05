from __future__ import annotations

import os
import shutil

from .common import BuildConfig, append_path_file, write_env_file


def prepare(config: BuildConfig) -> None:
    config.export_runtime_environment()
    if config.run_root.exists():
        shutil.rmtree(config.run_root)
    for path in [config.src_dir, config.build_dir, config.install_dir, config.artifacts_dir, config.ccache_dir]:
        path.mkdir(parents=True, exist_ok=True)

    write_env_file(
        os.environ.get("GITHUB_ENV"),
        {
            "RUN_ROOT": str(config.run_root),
            "SRC_DIR": str(config.src_dir),
            "BUILD_DIR": str(config.build_dir),
            "INSTALL_DIR": str(config.install_dir),
            "ARTIFACTS_DIR": str(config.artifacts_dir),
            "PACKAGE_NAME": config.package_name,
            "PACKAGE_ROOT": str(config.package_root),
            "ZIP_PATH": str(config.zip_path),
            "ANDROID_NDK": str(config.android_ndk),
            "LLVM_PREBUILT_DIR": str(config.llvm_prebuilt_dir),
            "CCACHE_DIR": str(config.ccache_dir),
        },
    )
    append_path_file(os.environ.get("GITHUB_PATH"), config.llvm_prebuilt_dir / "bin")

    print(f"RUN_ROOT={config.run_root}")
    print(f"ANDROID_NDK={config.android_ndk}")
    print(f"LLVM_PREBUILT_DIR={config.llvm_prebuilt_dir}")

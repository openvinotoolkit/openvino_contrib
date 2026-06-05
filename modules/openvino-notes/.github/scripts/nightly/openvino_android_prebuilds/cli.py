from __future__ import annotations

from enum import Enum
from typing import Annotated

import typer

from .apk_release import (
    package_release_apk_from_env,
    plan_apk_release_matrix_from_env,
    publish_apk_release_from_env,
)
from .build_steps import (
    build_genai_java_api,
    build_onetbb,
    build_openvino_genai,
    build_openvino_java_api,
    build_openvino_runtime,
    configure_openvino,
    install_openvino,
)
from .common import BuildConfig
from .package import ccache_stats, package_prebuild
from .release import publish_release_from_env
from .sources import checkout_sources, record_source_manifest
from .workspace import prepare


class Stage(str, Enum):
    all = "all"
    prepare = "prepare"
    checkout_sources = "checkout-sources"
    record_source_manifest = "record-source-manifest"
    build_onetbb = "build-onetbb"
    configure_openvino = "configure-openvino"
    build_openvino_runtime = "build-openvino-runtime"
    build_openvino_genai = "build-openvino-genai"
    build_openvino_java_api = "build-openvino-java-api"
    install_openvino = "install-openvino"
    build_genai_java_api = "build-genai-java-api"
    package_prebuild = "package-prebuild"
    ccache_stats = "ccache-stats"
    publish_release = "publish-release"
    plan_apk_release_matrix = "plan-apk-release-matrix"
    package_release_apk = "package-release-apk"
    publish_apk_release = "publish-apk-release"


STAGES = {
    Stage.prepare: prepare,
    Stage.checkout_sources: checkout_sources,
    Stage.record_source_manifest: record_source_manifest,
    Stage.build_onetbb: build_onetbb,
    Stage.configure_openvino: configure_openvino,
    Stage.build_openvino_runtime: build_openvino_runtime,
    Stage.build_openvino_genai: build_openvino_genai,
    Stage.build_openvino_java_api: build_openvino_java_api,
    Stage.install_openvino: install_openvino,
    Stage.build_genai_java_api: build_genai_java_api,
    Stage.package_prebuild: package_prebuild,
    Stage.ccache_stats: ccache_stats,
}


def run_all(config: BuildConfig) -> None:
    for stage, action in STAGES.items():
        if stage == Stage.ccache_stats:
            continue
        action(config)
    ccache_stats(config)


def run_stage(stage: Annotated[Stage, typer.Argument(help="Build stage to execute.")] = Stage.all) -> None:
    if stage == Stage.publish_release:
        publish_release_from_env()
        return
    if stage == Stage.plan_apk_release_matrix:
        plan_apk_release_matrix_from_env()
        return
    if stage == Stage.package_release_apk:
        package_release_apk_from_env()
        return
    if stage == Stage.publish_apk_release:
        publish_apk_release_from_env()
        return

    config = BuildConfig.from_env()
    if stage == Stage.all:
        run_all(config)
        return

    STAGES[stage](config)


def main() -> None:
    typer.run(run_stage)

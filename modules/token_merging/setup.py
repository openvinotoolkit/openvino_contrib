# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from setuptools import find_packages, setup

EXTRAS_REQUIRE = {
    "tests": ["onnx", "onnxruntime", "accelerate", "diffusers", "openvino", "optimum", "optimum-intel", "open-clip-torch","timm", "pytest"],
}

setup(
    name="tomeov",
    version="0.1.0",
    author="Alexander Kozlov",
    url="https://github.com/openvinotoolkit/openvino_contrib/tree/master/modules/token_merging",
    description="Token Merging for OpenVINO",
    install_requires=["torch~=2.6", "torchvision~=0.21"],
    dependency_links=["https://download.pytorch.org/whl/cpu"],
    extras_require=EXTRAS_REQUIRE,
    packages=find_packages(exclude=("examples", "build")),
    license = 'Apache 2.0',
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
)
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
from setuptools import setup, find_packages

EXTRAS_REQUIRE = {
    "benchmarks": [
        "datasets==2.14.7",
        "rouge==1.0.1",
        "scikit-learn>=1.7"
    ],
}

INSTALL_REQUIRES = [
    "torch==2.8.0",
    "torchvision==0.23.0",
    "transformers>=4.48.0",
    "accelerate==1.9.0",
]

# Safely read README.md for PyPI long description
path = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(path, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="genai_opt",
    version="0.1.0",
    author="Liubov Talamanova",
    url="https://github.com/openvinotoolkit/openvino_contrib/tree/master/modules/genai_optimizations",
    description="GenAI Inference Optimizations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="Apache-2.0",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
)

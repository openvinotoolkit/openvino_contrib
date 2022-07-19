# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import re
import os
from setuptools import find_namespace_packages, setup

# Ensure we match the version set in src/optimum/version.py
try:
    filepath = "optimum/version.py"
    with open(filepath) as version_file:
        (__version__,) = re.findall('__version__ = "(.*)"', version_file.read())
except Exception as error:
    raise Exception(f"Error: Could not open '{filepath}' due {error}\n")


install_requires = [
    "transformers",
    "openvino",
    "ovmsclient",
    "openmodelzoo-modelapi @ git+https://github.com/mzegla/open_model_zoo.git@adapters_changes#egg=openmodelzoo-modelapi&subdirectory=demos/common/python",  # noqa
]

nncf_deps = ["openvino-dev[onnx]", "nncf", "transformers<4.16.0", "datasets"]

# Add patches as data
folder = "optimum/intel/nncf/patches"
data = [os.path.join(folder, name) for name in os.listdir(folder)]

setup(
    name="openvino-optimum",
    version=__version__,
    description="Intel OpenVINO extension for Hugging Face Transformers",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: Apache Software License",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="transformers, quantization, pruning, training, intel, openvino",
    url="https://github.com/openvinotoolkit/openvino_contrib/",
    author="Intel Corporation",
    author_email="openvino_pushbot@intel.com",
    license="Apache",
    packages=find_namespace_packages(include=["optimum.*"]),
    install_requires=install_requires,
    extras_require={"nncf": nncf_deps, "all": nncf_deps},
    data_files=[("../../optimum/intel/nncf/patches", data)],
)

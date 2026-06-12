# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import sys
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension
import glob

_ext_src_root = "_ext_src"

def check_cuda_available():
    """Check if CUDA is available"""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

def get_extension_modules():
    """Return appropriate extension modules based on CUDA availability"""
    if check_cuda_available():
        # CUDA available, build CUDA version
        print("CUDA detected, building CUDA extensions...")
        _ext_sources = glob.glob("{}/src/*.cpp".format(_ext_src_root)) + glob.glob(
            "{}/src/*.cu".format(_ext_src_root)
        )
        return [
            CUDAExtension(
                name='pointnet2._ext',
                sources=_ext_sources,
                include_dirs=[os.path.join(_ext_src_root, "include")],
                extra_compile_args={
                    "cxx": ["-DCUDA_AVAILABLE=1"],
                    "nvcc": ["-O3", 
                        "-DCUDA_HAS_FP16=1",
                        "-D__CUDA_NO_HALF_OPERATORS__",
                        "-D__CUDA_NO_HALF_CONVERSIONS__",
                        "-D__CUDA_NO_HALF2_OPERATORS__",
                        "-DCUDA_AVAILABLE=1"
                    ]
                }
            )
        ]
    else:
        # CUDA not available, build CPU-only version
        print("CUDA not available, building CPU-only extensions...")
        _ext_sources = glob.glob("{}/src/*.cpp".format(_ext_src_root))
        return [
            CppExtension(
                name='pointnet2._ext',
                sources=_ext_sources,
                include_dirs=[os.path.join(_ext_src_root, "include")],
                extra_compile_args={
                    "cxx": ["-O3", "-DCUDA_AVAILABLE=0"]
                }
            )
        ]

setup(
    name='pointnet2',
    packages=find_packages(),
    ext_modules=get_extension_modules(),
    cmdclass={'build_ext': BuildExtension.with_options(use_ninja=False)}
)

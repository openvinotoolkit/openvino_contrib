# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
OpenVINO GPU Extensions for HGGD Point Cloud Operations

This package provides GPU-accelerated point cloud operations using
OpenVINO custom C++ extensions.

Usage:
    # Install pytorch3d shim
    from ov_gpu_extensions.ov_shim_v3 import install_ov_shim
    install_ov_shim()
    
    # Or use ops directly
    from ov_gpu_extensions.pointcloud_ops_native import create_ops
    ops = create_ops(device='CPU')
"""

from .pointcloud_ops_native_gpu import NativePointCloudOpsGPU
from .ov_shim_gpu import install_ov_shim

__all__ = ['NativePointCloudOpsGPU', 'install_ov_shim']

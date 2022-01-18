# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(ARM_COMPUTE_INCLUDE_DIR "" CACHE PATH "Path to custom ARM ComputeLibrary headers")
set(ARM_COMPUTE_LIB_DIR "" CACHE PATH "Path to custom ARM ComputeLibrary libraries")
set(ARM_COMPUTE_TOOLCHAIN_PREFIX "" CACHE PATH "Toolchain prefix for cross-compilation")

if(ARM)
    set(ARM_COMPUTE_TARGET_ARCH_DEFAULT armv7a)
    set(ARM_COMPUTE_TARGET_ARCHS armv7a)
elseif(AARCH64)
    if(APPLE)
        # Apple M1 is assumed
        set(ARM_COMPUTE_TARGET_ARCH_DEFAULT armv8.6-a)
    else()
        set(ARM_COMPUTE_TARGET_ARCH_DEFAULT arm64-v8a)
    endif()
    set(ARM_COMPUTE_TARGET_ARCHS arm64-v8a arm64-v8.2-a arm64-v8.2-a-sve arm64-v8.2-a-sve2
                                 armv8a armv8.2-a armv8.2-a-sve armv8.6-a armv8.6-a-sve armv8.6-a-sve2
                                 armv8r64)
endif()

set(ARM_COMPUTE_TARGET_ARCH "${ARM_COMPUTE_TARGET_ARCH_DEFAULT}" CACHE STRING "Architecture for ARM ComputeLibrary")
set_property(CACHE ARM_COMPUTE_TARGET_ARCH PROPERTY STRINGS ${ARM_COMPUTE_TARGET_ARCHS})

set(ARM_COMPUTE_SCONS_JOBS "" CACHE STRING "Number of simultaneous jobs to build ARM ComputeLibrary with")

# Print features

set(IE_OPTIONS ARM_COMPUTE_INCLUDE_DIR ARM_COMPUTE_LIB_DIR
               ARM_COMPUTE_TOOLCHAIN_PREFIX ARM_COMPUTE_TARGET_ARCH
               ARM_COMPUTE_SCONS_JOBS)

print_enabled_features()

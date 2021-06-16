# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(ARM_COMPUTE_INCLUDE_DIR OR ARM_COMPUTE_LIB_DIR)
    if (NOT ARM_COMPUTE_INCLUDE_DIR)
        message(FATAL_ERROR "Undefined ARM_COMPUTE_INCLUDE_DIR input variable should be set manually")
    else()
        message(STATUS "Using ${ARM_COMPUTE_INCLUDE_DIR} to include arm compute library headers")
    endif()

    if (NOT ARM_COMPUTE_LIB_DIR)
        message(FATAL_ERROR "Undefined ARM_COMPUTE_LIB_DIR input variable should be set manually")
    else()
        find_library(
            ARM_COMPUTE_LIB
            arm_compute-static
            PATHS ${ARM_COMPUTE_LIB_DIR}
        )
        message(STATUS "Found arm_compute-static: ${ARM_COMPUTE_LIB}")
        add_library(arm_compute STATIC IMPORTED GLOBAL)
        set_target_properties(arm_compute PROPERTIES
            IMPORTED_LOCATION ${ARM_COMPUTE_LIB}
            INTERFACE_INCLUDE_DIRECTORIES ${ARM_COMPUTE_INCLUDE_DIR})
        find_library(
            ARM_COMPUTE_CORE_LIB
            arm_compute_core-static
            PATHS ${ARM_COMPUTE_LIB_DIR}
        )
        message(STATUS "Found arm_compute_core-static: ${ARM_COMPUTE_CORE_LIB}")
        add_library(arm_compute_core STATIC IMPORTED GLOBAL)
        set_target_properties(arm_compute_core PROPERTIES
            IMPORTED_LOCATION ${ARM_COMPUTE_CORE_LIB}
            INTERFACE_INCLUDE_DIRECTORIES ${ARM_COMPUTE_INCLUDE_DIR})
    endif()

    add_library(half INTERFACE IMPORTED GLOBAL)
    set_target_properties(half PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES  ${ARM_COMPUTE_INCLUDE_DIR})
else()
    message(STATUS "Configure to build ${CMAKE_CURRENT_SOURCE_DIR}/ComputeLibrary")

    find_host_program(SCONS
        scons
    )

    if (NOT SCONS)
        message(FATAL_ERROR "Scons tool is not found!")
    endif()

    set(ARM_COMPURE_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/ComputeLibrary)
    file(GLOB_RECURSE SOURCES
        ${ARM_COMPURE_SOURCE_DIR}/*.cpp
        ${ARM_COMPURE_SOURCE_DIR}/*.hpp
        ${ARM_COMPURE_SOURCE_DIR}/*.h
    )

    set(extra_cxx_flags "-fPIC ${CMAKE_CXX_FLAGS} -Wno-undef")

    set(ARM_COMPUTE_OPTIONS
        neon=1
        opencl=0
        cppthreads=1
        embed_kernels=0
        examples=0
        internal_only=0
        Werror=0
        data_layout_support=nchw
        build_dir=${CMAKE_CURRENT_BINARY_DIR}
        arch=${ARM_COMPUTE_TARGET_ARCH}
    )

    if (ARM_COMPUTE_SCONS_JOBS)
        list(APPEND ARM_COMPUTE_OPTIONS -j${ARM_COMPUTE_SCONS_JOBS})
    endif()
    
    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
        list(APPEND ARM_COMPUTE_OPTIONS debug=1)
    endif()

    if(ANDROID)
        list(APPEND ARM_COMPUTE_OPTIONS os=android)
    elseif(APPLE)
        list(APPEND ARM_COMPUTE_OPTIONS os=macos)
    endif()

    if(NOT CMAKE_CROSSCOMPILING)
        list(APPEND ARM_COMPUTE_OPTIONS build=native)
    endif()

    if (CMAKE_CXX_COMPILER_LAUNCHER)
        list(APPEND ARM_COMPUTE_OPTIONS compiler_cache=${CMAKE_CXX_COMPILER_LAUNCHER})
    endif()

    # used to build for yocto
    if (ARM_COMPUTE_TOOLCHAIN_PREFIX)
        list(APPEND ARM_COMPUTE_OPTIONS toolchain_prefix=${ARM_COMPUTE_TOOLCHAIN_PREFIX})
    endif()

    if(ANDROID)
        if(ANDROID_PLATFORM_LEVEL LESS 18)
            message(FATAL_ERROR "ARM compute library requires Android API 18 level and higher"
                                "Please, speficy -DANDROID_PLATFORM=android-18 at least")
        endif()

        list(APPEND ARM_COMPUTE_OPTIONS
            toolchain_prefix="${ANDROID_TOOLCHAIN_PREFIX}"
            compiler_prefix="${ANDROID_TOOLCHAIN_ROOT}/bin/")

        set(extra_flags "${extra_flags} --target=${ANDROID_LLVM_TRIPLE}")
        set(extra_flags "${extra_flags} --gcc-toolchain=${ANDROID_TOOLCHAIN_ROOT}")
        set(extra_flags "${extra_flags} --sysroot=${CMAKE_SYSROOT}")

        set(extra_link_flags "${extra_link_flags} ${extra_flags}")
        set(extra_cxx_flags "${extra_cxx_flags} ${extra_flags}")
    elseif(APPLE)
        set(extra_cxx_flags "${extra_cxx_flags} -Wno-error=return-stack-address")
    endif()

    if(ENABLE_LTO)
        if((CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR
            CMAKE_CXX_COMPILER_ID MATCHES "^(Apple)?Clang$")
           AND (NOT CMAKE_CROSSCOMPILING))
            set(extra_cxx_flags "${extra_cxx_flags} -flto=thin")
            set(extra_link_flags "${extra_link_flags} -flto=thin")
        endif()
    endif()

    if(extra_link_flags)
        list(APPEND ARM_COMPUTE_OPTIONS extra_link_flags=${extra_link_flags})
    endif()

    list(APPEND ARM_COMPUTE_OPTIONS extra_cxx_flags=${extra_cxx_flags})

    add_custom_command(
        OUTPUT
            ${CMAKE_CURRENT_BINARY_DIR}/libarm_compute-static.a
            ${CMAKE_CURRENT_BINARY_DIR}/libarm_compute_core-static.a
        COMMAND ${CMAKE_COMMAND} -E env ${SCONS} ${ARM_COMPUTE_OPTIONS}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/ComputeLibrary
        COMMENT "Build Arm Compute Library"
        DEPENDS ${SOURCES}
                ${CMAKE_CURRENT_LIST_FILE}
                ${ARM_COMPURE_SOURCE_DIR}/SConscript
                ${ARM_COMPURE_SOURCE_DIR}/SConstruct
                )

    # Import targets

    add_custom_target(arm_compute_static_libs
        DEPENDS
            ${CMAKE_CURRENT_BINARY_DIR}/libarm_compute-static.a
            ${CMAKE_CURRENT_BINARY_DIR}/libarm_compute_core-static.a
    )

    add_library(arm_compute::arm_compute STATIC IMPORTED GLOBAL)
    set_target_properties(arm_compute::arm_compute PROPERTIES
        IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/libarm_compute-static.a
        INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}/ComputeLibrary)
    add_dependencies(arm_compute::arm_compute arm_compute_static_libs)

    add_library(arm_compute::arm_compute_core STATIC IMPORTED GLOBAL)
    set_target_properties(arm_compute::arm_compute_core PROPERTIES
        IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/libarm_compute_core-static.a
        INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}/ComputeLibrary)
    add_dependencies(arm_compute::arm_compute_core arm_compute_static_libs)

    add_library(arm_compute::half INTERFACE IMPORTED GLOBAL)
    set_target_properties(arm_compute::half PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}/ComputeLibrary/include)
endif()

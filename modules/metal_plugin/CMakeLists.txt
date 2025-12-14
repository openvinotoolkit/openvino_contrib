# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.13)

project(OpenVINOMetalPlugin)

find_package(OpenVINODeveloperPackage REQUIRED)

option(ENABLE_METAL_MLIR "Build MLIR-based METAL backend" ON)

if(ENABLE_METAL_MLIR)
    include(ExternalProject)

    set(METAL_LLVM_PROJECT_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/llvm-project")
    set(METAL_LLVM_PROJECT_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/third_party/llvm-project")
    set(METAL_LLVM_INSTALL_DIR "${METAL_LLVM_PROJECT_BINARY_DIR}/install")

    set(_metal_llvm_byproducts
        ${METAL_LLVM_INSTALL_DIR}/lib/libMLIRIR.a
        ${METAL_LLVM_INSTALL_DIR}/lib/libMLIRSupport.a
        ${METAL_LLVM_INSTALL_DIR}/lib/libMLIRPass.a
        ${METAL_LLVM_INSTALL_DIR}/lib/libMLIRTransforms.a
        ${METAL_LLVM_INSTALL_DIR}/lib/libMLIRBufferizationDialect.a
        ${METAL_LLVM_INSTALL_DIR}/lib/libMLIRBufferizationTransforms.a
        ${METAL_LLVM_INSTALL_DIR}/lib/libMLIRFuncTransforms.a
        ${METAL_LLVM_INSTALL_DIR}/lib/libMLIRMemRefTransforms.a
        ${METAL_LLVM_INSTALL_DIR}/lib/libMLIRFuncDialect.a
        ${METAL_LLVM_INSTALL_DIR}/lib/libMLIRLinalgDialect.a
        ${METAL_LLVM_INSTALL_DIR}/lib/libMLIRLinalgTransforms.a
        ${METAL_LLVM_INSTALL_DIR}/lib/libMLIRTensorDialect.a
        ${METAL_LLVM_INSTALL_DIR}/lib/libMLIRArithDialect.a
        ${METAL_LLVM_INSTALL_DIR}/lib/libMLIRSCFDialect.a
        ${METAL_LLVM_INSTALL_DIR}/lib/libLLVMSupport.a)

    ExternalProject_Add(metal_llvm_external
        SOURCE_DIR ${METAL_LLVM_PROJECT_SOURCE_DIR}/llvm
        BINARY_DIR ${METAL_LLVM_PROJECT_BINARY_DIR}/llvm
        CMAKE_ARGS
            -G ${CMAKE_GENERATOR}
            -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
            -DCMAKE_OSX_ARCHITECTURES=${CMAKE_OSX_ARCHITECTURES}
            -DCMAKE_INSTALL_PREFIX=${METAL_LLVM_INSTALL_DIR}
            -DLLVM_ENABLE_PROJECTS=mlir
            -DLLVM_TARGETS_TO_BUILD=Native
            -DBUILD_SHARED_LIBS=OFF
            -DLLVM_BUILD_LLVM_DYLIB=OFF
            -DLLVM_ENABLE_BINDINGS=OFF
            -DLLVM_INCLUDE_EXAMPLES=OFF
            -DLLVM_INCLUDE_TESTS=OFF
            -DMLIR_ENABLE_BINDINGS_PYTHON=OFF
            -DLLVM_ENABLE_RTTI=ON
        BUILD_BYPRODUCTS ${_metal_llvm_byproducts}
        INSTALL_COMMAND ${CMAKE_COMMAND} --build . --target install)

    # Try to use installed config packages from the external build (if already present).
    set(MLIR_DIR "${METAL_LLVM_INSTALL_DIR}/lib/cmake/mlir" CACHE PATH "MLIR config dir for METAL" FORCE)
    set(LLVM_DIR "${METAL_LLVM_INSTALL_DIR}/lib/cmake/llvm" CACHE PATH "LLVM config dir for METAL" FORCE)

    find_package(LLVM CONFIG PATHS ${LLVM_DIR} NO_DEFAULT_PATH QUIET)
    find_package(MLIR CONFIG PATHS ${MLIR_DIR} NO_DEFAULT_PATH QUIET)

    if(LLVM_FOUND AND MLIR_FOUND)
        message(STATUS "Using external LLVM/MLIR from ${METAL_LLVM_INSTALL_DIR}")
        set(MLIR_INCLUDE_DIRS
            ${MLIR_INCLUDE_DIRS}
            ${LLVM_INCLUDE_DIRS}
            CACHE INTERNAL "MLIR include directories for METAL plugin")
    else()
        message(STATUS "LLVM/MLIR not found yet; will use ExternalProject outputs directly")
        file(MAKE_DIRECTORY ${METAL_LLVM_INSTALL_DIR}/include)
        file(MAKE_DIRECTORY ${METAL_LLVM_INSTALL_DIR}/lib)
        set(MLIR_INCLUDE_DIRS
            ${METAL_LLVM_INSTALL_DIR}/include
            CACHE INTERNAL "MLIR include directories for METAL plugin")

        function(_metal_import_lib name)
            add_library(${name} STATIC IMPORTED GLOBAL)
            set_target_properties(${name} PROPERTIES
                IMPORTED_LOCATION "${METAL_LLVM_INSTALL_DIR}/lib/lib${name}.a"
                INTERFACE_INCLUDE_DIRECTORIES "${MLIR_INCLUDE_DIRS}")
            add_dependencies(${name} metal_llvm_external)
        endfunction()

        _metal_import_lib(MLIRIR)
        _metal_import_lib(MLIRSupport)
        _metal_import_lib(MLIRPass)
        _metal_import_lib(MLIRTransforms)
        _metal_import_lib(MLIRBufferizationDialect)
        _metal_import_lib(MLIRBufferizationTransforms)
        _metal_import_lib(MLIRFuncTransforms)
        _metal_import_lib(MLIRMemRefTransforms)
        _metal_import_lib(MLIRFuncDialect)
        _metal_import_lib(MLIRLinalgDialect)
        _metal_import_lib(MLIRLinalgTransforms)
        _metal_import_lib(MLIRTensorDialect)
        _metal_import_lib(MLIRArithDialect)
        _metal_import_lib(MLIRSCFDialect)
        _metal_import_lib(LLVMSupport)
    endif()

    # propagate dependency to subtargets
    set(METAL_LLVM_EXTERNAL_TARGET metal_llvm_external CACHE INTERNAL "External llvm-project target for METAL plugin")
endif()

ov_option(ENABLE_METAL_REGISTRATION "Enables registration of METAL plugin" ON)

if(CMAKE_COMPILER_IS_GNUCXX)
    ov_add_compiler_flags(-Wall)
endif()

add_subdirectory(src)

if(ENABLE_TESTS)
    add_subdirectory(tests)
endif()

# install packaging (re-uses Dev Package helper)
if(OpenVINODeveloperPackage_FOUND)
    ov_cpack(metal)
endif()

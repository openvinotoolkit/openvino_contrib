# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.13)

project(OpenVINOGfxPlugin)

find_package(OpenVINODeveloperPackage REQUIRED)

option(ENABLE_GFX_MLIR "Build MLIR-based GFX backend" ON)

if(ENABLE_GFX_MLIR)
    include(ExternalProject)

    set(GFX_LLVM_PROJECT_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/llvm-project")
    set(GFX_LLVM_PROJECT_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/third_party/llvm-project")
    set(GFX_LLVM_INSTALL_DIR "${GFX_LLVM_PROJECT_BINARY_DIR}/install")

    set(_gfx_llvm_byproducts
        ${GFX_LLVM_INSTALL_DIR}/lib/libMLIRIR.a
        ${GFX_LLVM_INSTALL_DIR}/lib/libMLIRSupport.a
        ${GFX_LLVM_INSTALL_DIR}/lib/libLLVMDemangle.a
        ${GFX_LLVM_INSTALL_DIR}/lib/libMLIRPass.a
        ${GFX_LLVM_INSTALL_DIR}/lib/libMLIRTransforms.a
        ${GFX_LLVM_INSTALL_DIR}/lib/libMLIRBufferizationDialect.a
        ${GFX_LLVM_INSTALL_DIR}/lib/libMLIRBufferizationTransforms.a
        ${GFX_LLVM_INSTALL_DIR}/lib/libMLIRFuncTransforms.a
        ${GFX_LLVM_INSTALL_DIR}/lib/libMLIRMemRefTransforms.a
        ${GFX_LLVM_INSTALL_DIR}/lib/libMLIRFuncDialect.a
        ${GFX_LLVM_INSTALL_DIR}/lib/libMLIRLinalgDialect.a
        ${GFX_LLVM_INSTALL_DIR}/lib/libMLIRLinalgTransforms.a
        ${GFX_LLVM_INSTALL_DIR}/lib/libMLIRLinalgUtils.a
        ${GFX_LLVM_INSTALL_DIR}/lib/libMLIRCallInterfaces.a
        ${GFX_LLVM_INSTALL_DIR}/lib/libMLIRValueBoundsOpInterface.a
        ${GFX_LLVM_INSTALL_DIR}/lib/libMLIRAnalysis.a
        ${GFX_LLVM_INSTALL_DIR}/lib/libMLIRArithTransforms.a
        ${GFX_LLVM_INSTALL_DIR}/lib/libMLIRComplexDialect.a
        ${GFX_LLVM_INSTALL_DIR}/lib/libMLIRMathDialect.a
        ${GFX_LLVM_INSTALL_DIR}/lib/libMLIRMathTransforms.a
        ${GFX_LLVM_INSTALL_DIR}/lib/libMLIRSparseTensorDialect.a
        ${GFX_LLVM_INSTALL_DIR}/lib/libMLIRSparseTensorTransforms.a
        ${GFX_LLVM_INSTALL_DIR}/lib/libMLIRTensorDialect.a
        ${GFX_LLVM_INSTALL_DIR}/lib/libMLIRArithDialect.a
        ${GFX_LLVM_INSTALL_DIR}/lib/libMLIRSCFDialect.a
        ${GFX_LLVM_INSTALL_DIR}/lib/libLLVMSupport.a)

    ExternalProject_Add(gfx_llvm_external
        SOURCE_DIR ${GFX_LLVM_PROJECT_SOURCE_DIR}/llvm
        BINARY_DIR ${GFX_LLVM_PROJECT_BINARY_DIR}/llvm
        CMAKE_ARGS
            -G ${CMAKE_GENERATOR}
            -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
            -DCMAKE_OSX_ARCHITECTURES=${CMAKE_OSX_ARCHITECTURES}
            -DCMAKE_INSTALL_PREFIX=${GFX_LLVM_INSTALL_DIR}
            -DLLVM_ENABLE_PROJECTS=mlir
            -DLLVM_TARGETS_TO_BUILD=Native
            -DBUILD_SHARED_LIBS=OFF
            -DLLVM_BUILD_LLVM_DYLIB=OFF
            -DLLVM_BUILD_TOOLS=OFF
            -DLLVM_BUILD_UTILS=OFF
            -DLLVM_ENABLE_BINDINGS=OFF
            -DLLVM_INCLUDE_EXAMPLES=OFF
            -DLLVM_INCLUDE_TESTS=OFF
            -DMLIR_BUILD_TOOLS=OFF
            -DMLIR_INCLUDE_TOOLS=OFF
            -DMLIR_INCLUDE_TESTS=OFF
            -DMLIR_ENABLE_BINDINGS_PYTHON=OFF
            -DLLVM_ENABLE_RTTI=ON
        BUILD_BYPRODUCTS ${_gfx_llvm_byproducts}
        INSTALL_COMMAND ${CMAKE_COMMAND} --build . --target install)

    # Try to use installed config packages from the external build (if already present).
    set(MLIR_DIR "${GFX_LLVM_INSTALL_DIR}/lib/cmake/mlir" CACHE PATH "MLIR config dir for GFX" FORCE)
    set(LLVM_DIR "${GFX_LLVM_INSTALL_DIR}/lib/cmake/llvm" CACHE PATH "LLVM config dir for GFX" FORCE)

    find_package(LLVM CONFIG PATHS ${LLVM_DIR} NO_DEFAULT_PATH QUIET)
    find_package(MLIR CONFIG PATHS ${MLIR_DIR} NO_DEFAULT_PATH QUIET)

    if(LLVM_FOUND AND MLIR_FOUND)
        message(STATUS "Using external LLVM/MLIR from ${GFX_LLVM_INSTALL_DIR}")
        set(MLIR_INCLUDE_DIRS
            ${MLIR_INCLUDE_DIRS}
            ${LLVM_INCLUDE_DIRS}
            CACHE INTERNAL "MLIR include directories for GFX plugin")
    else()
        message(STATUS "LLVM/MLIR not found yet; will use ExternalProject outputs directly")
        file(MAKE_DIRECTORY ${GFX_LLVM_INSTALL_DIR}/include)
        file(MAKE_DIRECTORY ${GFX_LLVM_INSTALL_DIR}/lib)
        set(MLIR_INCLUDE_DIRS
            ${GFX_LLVM_INSTALL_DIR}/include
            CACHE INTERNAL "MLIR include directories for GFX plugin")

        function(_gfx_import_lib name)
            add_library(${name} STATIC IMPORTED GLOBAL)
            set_target_properties(${name} PROPERTIES
                IMPORTED_LOCATION "${GFX_LLVM_INSTALL_DIR}/lib/lib${name}.a"
                INTERFACE_INCLUDE_DIRECTORIES "${MLIR_INCLUDE_DIRS}")
            add_dependencies(${name} gfx_llvm_external)
        endfunction()

        _gfx_import_lib(MLIRIR)
        _gfx_import_lib(MLIRSupport)
        _gfx_import_lib(LLVMDemangle)
        _gfx_import_lib(MLIRPass)
        _gfx_import_lib(MLIRTransforms)
        _gfx_import_lib(MLIRBufferizationDialect)
        _gfx_import_lib(MLIRBufferizationTransforms)
        _gfx_import_lib(MLIRFuncTransforms)
        _gfx_import_lib(MLIRMemRefTransforms)
        _gfx_import_lib(MLIRFuncDialect)
        _gfx_import_lib(MLIRLinalgDialect)
        _gfx_import_lib(MLIRLinalgTransforms)
        _gfx_import_lib(MLIRLinalgUtils)
        _gfx_import_lib(MLIRCallInterfaces)
        _gfx_import_lib(MLIRValueBoundsOpInterface)
        _gfx_import_lib(MLIRAnalysis)
        _gfx_import_lib(MLIRArithTransforms)
        _gfx_import_lib(MLIRComplexDialect)
        _gfx_import_lib(MLIRMathDialect)
        _gfx_import_lib(MLIRMathTransforms)
        _gfx_import_lib(MLIRSparseTensorDialect)
        _gfx_import_lib(MLIRSparseTensorTransforms)
        _gfx_import_lib(MLIRTensorDialect)
        _gfx_import_lib(MLIRArithDialect)
        _gfx_import_lib(MLIRSCFDialect)
        _gfx_import_lib(LLVMSupport)
    endif()

    # propagate dependency to subtargets
    set(GFX_LLVM_EXTERNAL_TARGET gfx_llvm_external CACHE INTERNAL "External llvm-project target for GFX plugin")
endif()

ov_option(ENABLE_GFX_REGISTRATION "Enables registration of GFX plugin" ON)

if(CMAKE_COMPILER_IS_GNUCXX)
    ov_add_compiler_flags(-Wall)
endif()

add_subdirectory(src)

if(ENABLE_TESTS)
    add_subdirectory(tests)
endif()

# install packaging (re-uses Dev Package helper)
if(OpenVINODeveloperPackage_FOUND)
    ov_cpack(gfx)
endif()

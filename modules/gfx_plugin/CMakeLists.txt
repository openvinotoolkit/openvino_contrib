# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.13)

project(OpenVINOGfxPlugin)

find_package(OpenVINODeveloperPackage REQUIRED)

option(ENABLE_GFX_MLIR "Build MLIR-based GFX backend" ON)
if(NOT ENABLE_GFX_MLIR)
    message(FATAL_ERROR "GFX: MLIR is required for the GFX plugin.")
endif()
include(cmake/GfxSources.cmake)
include(cmake/GfxBackendConfig.cmake)

if(ENABLE_GFX_MLIR)
    include(ExternalProject)
    include(FindPkgConfig)

    set(GFX_LLVM_PROJECT_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/llvm-project")
    if(ANDROID)
        set(GFX_LLVM_PROJECT_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/third_party/llvm-project-android")
    else()
        set(GFX_LLVM_PROJECT_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/third_party/llvm-project")
    endif()
    set(GFX_LLVM_INSTALL_DIR "${GFX_LLVM_PROJECT_BINARY_DIR}/install")

    set(_gfx_llvm_toolchain_args "")
    if(ANDROID)
        list(APPEND _gfx_llvm_toolchain_args
            -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}
            -DANDROID_ABI=${ANDROID_ABI}
            -DANDROID_PLATFORM=${ANDROID_PLATFORM}
            -DANDROID_STL=${ANDROID_STL})
    endif()

    # Build products are shared libraries; explicit byproduct list is unnecessary.
    set(_gfx_llvm_byproducts)

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
            -DBUILD_SHARED_LIBS=ON
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
            ${_gfx_llvm_toolchain_args}
        INSTALL_COMMAND ${CMAKE_COMMAND} --build . --target install)

    # Try to use installed config packages from the external build (if already present).
    set(MLIR_DIR "${GFX_LLVM_INSTALL_DIR}/lib/cmake/mlir" CACHE PATH "MLIR config dir for GFX" FORCE)
    set(LLVM_DIR "${GFX_LLVM_INSTALL_DIR}/lib/cmake/llvm" CACHE PATH "LLVM config dir for GFX" FORCE)

    if(ANDROID)
        set(LLVM_FOUND OFF)
        set(MLIR_FOUND OFF)
    else()
        find_package(LLVM CONFIG PATHS ${LLVM_DIR} NO_DEFAULT_PATH QUIET)
        find_package(MLIR CONFIG PATHS ${MLIR_DIR} NO_DEFAULT_PATH QUIET)
    endif()

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
        _gfx_import_lib(MLIRControlFlowDialect)
        _gfx_import_lib(MLIRControlFlowInterfaces)
        _gfx_import_lib(MLIRIndexDialect)
        _gfx_import_lib(MLIRMemRefDialect)
        _gfx_import_lib(MLIRFunctionInterfaces)
        _gfx_import_lib(MLIRDestinationStyleOpInterface)
        _gfx_import_lib(MLIRShapedOpInterfaces)
        _gfx_import_lib(MLIRIndexingMapOpInterface)
        _gfx_import_lib(MLIRLoopLikeInterface)
        _gfx_import_lib(MLIRParallelCombiningOpInterface)
        _gfx_import_lib(MLIRMemorySlotInterfaces)
        _gfx_import_lib(MLIRCopyOpInterface)
        _gfx_import_lib(MLIRArithValueBoundsOpInterfaceImpl)
        _gfx_import_lib(MLIRTensorInferTypeOpInterfaceImpl)
        _gfx_import_lib(MLIRTensorTilingInterfaceImpl)
        _gfx_import_lib(MLIRInferIntRangeCommon)
        _gfx_import_lib(MLIRInferIntRangeInterface)
        _gfx_import_lib(MLIRInferTypeOpInterface)
        _gfx_import_lib(MLIRSideEffectInterfaces)
        _gfx_import_lib(MLIRViewLikeInterface)
        _gfx_import_lib(MLIRCastInterfaces)
        _gfx_import_lib(MLIRDataLayoutInterfaces)
        _gfx_import_lib(MLIRDialect)
        _gfx_import_lib(MLIRDialectUtils)
        _gfx_import_lib(MLIRTransformUtils)
        _gfx_import_lib(MLIRRewrite)
        _gfx_import_lib(MLIRParser)
        _gfx_import_lib(MLIRAsmParser)
        _gfx_import_lib(MLIRBytecodeReader)
        _gfx_import_lib(MLIRBytecodeWriter)
        _gfx_import_lib(MLIRBytecodeOpInterface)
        _gfx_import_lib(MLIRUBDialect)
        _gfx_import_lib(MLIRPDLToPDLInterp)
        _gfx_import_lib(MLIRPDLInterpDialect)
        _gfx_import_lib(MLIRPDLDialect)
        _gfx_import_lib(MLIRPDLLAST)
        _gfx_import_lib(MLIRRewritePDL)
        _gfx_import_lib(MLIRSubsetOpInterface)
        _gfx_import_lib(MLIRReconcileUnrealizedCasts)
        _gfx_import_lib(MLIRSPIRVImageInterfaces)
        _gfx_import_lib(MLIRSPIRVBinaryUtils)
        _gfx_import_lib(MLIRAffineTransforms)
        _gfx_import_lib(MLIRAffineToStandard)
        _gfx_import_lib(MLIRAffineUtils)
        _gfx_import_lib(MLIRAffineAnalysis)
        _gfx_import_lib(MLIRAffineDialect)
        _gfx_import_lib(MLIRTensorUtils)
        _gfx_import_lib(MLIRTensorTransforms)
        _gfx_import_lib(MLIRTensorTransformOps)
        _gfx_import_lib(MLIRPresburger)
        _gfx_import_lib(MLIRArithUtils)
        _gfx_import_lib(MLIRSCFTransforms)
        _gfx_import_lib(MLIRSCFUtils)
        _gfx_import_lib(MLIRMemRefUtils)
        _gfx_import_lib(LLVMSupport)
        _gfx_import_lib(MLIRVectorToSPIRV)
        _gfx_import_lib(MLIRVectorDialect)
        _gfx_import_lib(MLIRVectorInterfaces)
        _gfx_import_lib(MLIRDLTIDialect)
        _gfx_import_lib(MLIRGPUDialect)
        _gfx_import_lib(MLIRGPUTransforms)
        _gfx_import_lib(MLIRSCFToGPU)
        _gfx_import_lib(MLIRSPIRVDialect)
        _gfx_import_lib(MLIRSPIRVSerialization)
        _gfx_import_lib(MLIRSPIRVTransforms)
        _gfx_import_lib(MLIRSPIRVUtils)
        _gfx_import_lib(MLIRSPIRVConversion)
        _gfx_import_lib(MLIRGPUToSPIRV)
        _gfx_import_lib(MLIRArithToSPIRV)
        _gfx_import_lib(MLIRMathToSPIRV)
        _gfx_import_lib(MLIRSCFToSPIRV)
        _gfx_import_lib(MLIRTensorToSPIRV)
        _gfx_import_lib(MLIRMemRefToSPIRV)
        _gfx_import_lib(MLIRFuncToSPIRV)
        _gfx_import_lib(MLIRIndexToSPIRV)
        _gfx_import_lib(MLIRControlFlowToSPIRV)
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

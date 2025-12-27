# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.13)

project(OpenVINOGfxPlugin)

find_package(OpenVINODeveloperPackage REQUIRED)

option(ENABLE_GFX_MLIR "Build MLIR-based GFX backend" ON)
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
        ${GFX_LLVM_INSTALL_DIR}/lib/libMLIRControlFlowDialect.a
        ${GFX_LLVM_INSTALL_DIR}/lib/libMLIRControlFlowInterfaces.a
        ${GFX_LLVM_INSTALL_DIR}/lib/libMLIRIndexDialect.a
        ${GFX_LLVM_INSTALL_DIR}/lib/libMLIRMemRefDialect.a
        ${GFX_LLVM_INSTALL_DIR}/lib/libMLIRFunctionInterfaces.a
        ${GFX_LLVM_INSTALL_DIR}/lib/libMLIRDestinationStyleOpInterface.a
        ${GFX_LLVM_INSTALL_DIR}/lib/libMLIRShapedOpInterfaces.a
        ${GFX_LLVM_INSTALL_DIR}/lib/libMLIRIndexingMapOpInterface.a
        ${GFX_LLVM_INSTALL_DIR}/lib/libMLIRLoopLikeInterface.a
        ${GFX_LLVM_INSTALL_DIR}/lib/libMLIRParallelCombiningOpInterface.a
        ${GFX_LLVM_INSTALL_DIR}/lib/libMLIRMemorySlotInterfaces.a
        ${GFX_LLVM_INSTALL_DIR}/lib/libMLIRCopyOpInterface.a
        ${GFX_LLVM_INSTALL_DIR}/lib/libMLIRArithValueBoundsOpInterfaceImpl.a
        ${GFX_LLVM_INSTALL_DIR}/lib/libMLIRTensorInferTypeOpInterfaceImpl.a
        ${GFX_LLVM_INSTALL_DIR}/lib/libMLIRTensorTilingInterfaceImpl.a
        ${GFX_LLVM_INSTALL_DIR}/lib/libMLIRInferIntRangeCommon.a
        ${GFX_LLVM_INSTALL_DIR}/lib/libMLIRInferIntRangeInterface.a
        ${GFX_LLVM_INSTALL_DIR}/lib/libMLIRInferTypeOpInterface.a
        ${GFX_LLVM_INSTALL_DIR}/lib/libMLIRSideEffectInterfaces.a
        ${GFX_LLVM_INSTALL_DIR}/lib/libMLIRViewLikeInterface.a
        ${GFX_LLVM_INSTALL_DIR}/lib/libMLIRCastInterfaces.a
        ${GFX_LLVM_INSTALL_DIR}/lib/libMLIRDataLayoutInterfaces.a
        ${GFX_LLVM_INSTALL_DIR}/lib/libMLIRDialect.a
        ${GFX_LLVM_INSTALL_DIR}/lib/libMLIRDialectUtils.a
        ${GFX_LLVM_INSTALL_DIR}/lib/libMLIRTransformUtils.a
        ${GFX_LLVM_INSTALL_DIR}/lib/libMLIRRewrite.a
        ${GFX_LLVM_INSTALL_DIR}/lib/libMLIRParser.a
        ${GFX_LLVM_INSTALL_DIR}/lib/libMLIRAsmParser.a
        ${GFX_LLVM_INSTALL_DIR}/lib/libMLIRBytecodeReader.a
        ${GFX_LLVM_INSTALL_DIR}/lib/libMLIRBytecodeWriter.a
        ${GFX_LLVM_INSTALL_DIR}/lib/libMLIRBytecodeOpInterface.a
        ${GFX_LLVM_INSTALL_DIR}/lib/libMLIRUBDialect.a
        ${GFX_LLVM_INSTALL_DIR}/lib/libMLIRPDLToPDLInterp.a
        ${GFX_LLVM_INSTALL_DIR}/lib/libMLIRPDLInterpDialect.a
        ${GFX_LLVM_INSTALL_DIR}/lib/libMLIRPDLDialect.a
        ${GFX_LLVM_INSTALL_DIR}/lib/libMLIRPDLLAST.a
        ${GFX_LLVM_INSTALL_DIR}/lib/libMLIRRewritePDL.a
        ${GFX_LLVM_INSTALL_DIR}/lib/libMLIRSubsetOpInterface.a
        ${GFX_LLVM_INSTALL_DIR}/lib/libMLIRReconcileUnrealizedCasts.a
        ${GFX_LLVM_INSTALL_DIR}/lib/libMLIRSPIRVImageInterfaces.a
        ${GFX_LLVM_INSTALL_DIR}/lib/libMLIRSPIRVBinaryUtils.a
        ${GFX_LLVM_INSTALL_DIR}/lib/libMLIRAffineTransforms.a
        ${GFX_LLVM_INSTALL_DIR}/lib/libMLIRAffineToStandard.a
        ${GFX_LLVM_INSTALL_DIR}/lib/libMLIRAffineUtils.a
        ${GFX_LLVM_INSTALL_DIR}/lib/libMLIRAffineAnalysis.a
        ${GFX_LLVM_INSTALL_DIR}/lib/libMLIRAffineDialect.a
        ${GFX_LLVM_INSTALL_DIR}/lib/libMLIRTensorUtils.a
        ${GFX_LLVM_INSTALL_DIR}/lib/libMLIRTensorTransforms.a
        ${GFX_LLVM_INSTALL_DIR}/lib/libMLIRTensorTransformOps.a
        ${GFX_LLVM_INSTALL_DIR}/lib/libMLIRPresburger.a
        ${GFX_LLVM_INSTALL_DIR}/lib/libMLIRArithUtils.a
        ${GFX_LLVM_INSTALL_DIR}/lib/libMLIRSCFTransforms.a
        ${GFX_LLVM_INSTALL_DIR}/lib/libMLIRSCFUtils.a
        ${GFX_LLVM_INSTALL_DIR}/lib/libMLIRMemRefUtils.a
        ${GFX_LLVM_INSTALL_DIR}/lib/libLLVMSupport.a
        ${GFX_LLVM_INSTALL_DIR}/lib/libMLIRVectorToSPIRV.a
        ${GFX_LLVM_INSTALL_DIR}/lib/libMLIRVectorDialect.a
        ${GFX_LLVM_INSTALL_DIR}/lib/libMLIRVectorInterfaces.a
        ${GFX_LLVM_INSTALL_DIR}/lib/libMLIRDLTIDialect.a
        ${GFX_LLVM_INSTALL_DIR}/lib/libMLIRGPUDialect.a
        ${GFX_LLVM_INSTALL_DIR}/lib/libMLIRGPUTransforms.a
        ${GFX_LLVM_INSTALL_DIR}/lib/libMLIRSCFToGPU.a
        ${GFX_LLVM_INSTALL_DIR}/lib/libMLIRSPIRVDialect.a
        ${GFX_LLVM_INSTALL_DIR}/lib/libMLIRSPIRVSerialization.a
        ${GFX_LLVM_INSTALL_DIR}/lib/libMLIRSPIRVTransforms.a
        ${GFX_LLVM_INSTALL_DIR}/lib/libMLIRSPIRVUtils.a
        ${GFX_LLVM_INSTALL_DIR}/lib/libMLIRSPIRVConversion.a
        ${GFX_LLVM_INSTALL_DIR}/lib/libMLIRGPUToSPIRV.a
        ${GFX_LLVM_INSTALL_DIR}/lib/libMLIRArithToSPIRV.a
        ${GFX_LLVM_INSTALL_DIR}/lib/libMLIRMathToSPIRV.a
        ${GFX_LLVM_INSTALL_DIR}/lib/libMLIRSCFToSPIRV.a
        ${GFX_LLVM_INSTALL_DIR}/lib/libMLIRTensorToSPIRV.a
        ${GFX_LLVM_INSTALL_DIR}/lib/libMLIRMemRefToSPIRV.a
        ${GFX_LLVM_INSTALL_DIR}/lib/libMLIRFuncToSPIRV.a
        ${GFX_LLVM_INSTALL_DIR}/lib/libMLIRIndexToSPIRV.a
        ${GFX_LLVM_INSTALL_DIR}/lib/libMLIRControlFlowToSPIRV.a)

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
            ${_gfx_llvm_toolchain_args}
        BUILD_BYPRODUCTS ${_gfx_llvm_byproducts}
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

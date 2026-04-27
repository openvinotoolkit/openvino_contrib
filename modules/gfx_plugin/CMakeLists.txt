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
    set(GFX_LLVM_BUILD_DIR "${GFX_LLVM_PROJECT_BINARY_DIR}/llvm-static")
    set(GFX_LLVM_INSTALL_DIR "${GFX_LLVM_PROJECT_BINARY_DIR}/install-static")
    set(GFX_LLVM_COMPONENT_LIBS
        LLVMSupport
        LLVMDemangle)
    set(GFX_MLIR_COMPONENT_LIBS
        MLIRIR
        MLIRSupport
        MLIRPass
        MLIRTransforms
        MLIRBufferizationDialect
        MLIRBufferizationTransforms
        MLIRFuncTransforms
        MLIRMemRefTransforms
        MLIRFuncDialect
        MLIRLinalgDialect
        MLIRLinalgTransforms
        MLIRLinalgUtils
        MLIRGPUDialect
        MLIRGPUTransforms
        MLIRSCFToGPU
        MLIRVectorDialect
        MLIRVectorInterfaces
        MLIRMaskableOpInterface
        MLIRMaskingOpInterface
        MLIRSubsetOpInterface
        MLIRSCFTransforms
        MLIRDLTIDialect
        MLIRReconcileUnrealizedCasts
        MLIRCallInterfaces
        MLIRValueBoundsOpInterface
        MLIRAnalysis
        MLIRArithTransforms
        MLIRComplexDialect
        MLIRMathDialect
        MLIRMathTransforms
        MLIRSparseTensorDialect
        MLIRSparseTensorTransforms
        MLIRTensorDialect
        MLIRArithDialect
        MLIRSCFDialect
        MLIRControlFlowDialect
        MLIRControlFlowInterfaces
        MLIRIndexDialect
        MLIRMemRefDialect
        MLIRMemOpInterfaces
        MLIRFunctionInterfaces
        MLIRDestinationStyleOpInterface
        MLIRShapedOpInterfaces
        MLIRIndexingMapOpInterface
        MLIRLoopLikeInterface
        MLIRParallelCombiningOpInterface
        MLIRMemorySlotInterfaces
        MLIRArithValueBoundsOpInterfaceImpl
        MLIRTensorInferTypeOpInterfaceImpl
        MLIRTensorTilingInterfaceImpl
        MLIRInferIntRangeCommon
        MLIRInferIntRangeInterface
        MLIRInferTypeOpInterface
        MLIRSideEffectInterfaces
        MLIRViewLikeInterface
        MLIRCastInterfaces
        MLIRDataLayoutInterfaces
        MLIRDialect
        MLIRDialectUtils
        MLIRTransformUtils
        MLIRRewrite
        MLIRRewritePDL
        MLIRPDLToPDLInterp
        MLIRPDLInterpDialect
        MLIRPDLDialect
        MLIRPDLLAST
        MLIRParser
        MLIRAsmParser
        MLIRBytecodeReader
        MLIRBytecodeWriter
        MLIRBytecodeOpInterface
        MLIRUBDialect
        MLIRUBToSPIRV
        MLIRSPIRVImageInterfaces
        MLIRSPIRVBinaryUtils
        MLIRAffineTransforms
        MLIRAffineToStandard
        MLIRAffineUtils
        MLIRAffineAnalysis
        MLIRAffineDialect
        MLIRTensorUtils
        MLIRTensorTransforms
        MLIRTensorTransformOps
        MLIRPresburger
        MLIRArithUtils
        MLIRSCFUtils
        MLIRMemRefUtils
        MLIRVectorToSPIRV
        MLIRSPIRVDialect
        MLIRSPIRVSerialization
        MLIRSPIRVTransforms
        MLIRSPIRVUtils
        MLIRSPIRVConversion
        MLIRGPUToSPIRV
        MLIRArithToSPIRV
        MLIRMathToSPIRV
        MLIRSCFToSPIRV
        MLIRTensorToSPIRV
        MLIRMemRefToSPIRV
        MLIRFuncToSPIRV
        MLIRIndexToSPIRV
        MLIRControlFlowToSPIRV)

    set(_gfx_llvm_toolchain_args "")
    if(ANDROID)
        set(_gfx_llvm_targets_to_build "AArch64")
        list(APPEND _gfx_llvm_toolchain_args
            -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}
            -DANDROID_ABI=${ANDROID_ABI}
            -DANDROID_PLATFORM=${ANDROID_PLATFORM}
            -DANDROID_STL=${ANDROID_STL})
    elseif(APPLE)
        set(_gfx_llvm_targets_to_build "Native")
        set(_gfx_llvm_osx_sysroot "${CMAKE_OSX_SYSROOT}")
        if((NOT _gfx_llvm_osx_sysroot OR NOT EXISTS "${_gfx_llvm_osx_sysroot}"))
            find_program(_gfx_xcrun_EXECUTABLE xcrun)
            if(_gfx_xcrun_EXECUTABLE)
                execute_process(
                    COMMAND "${_gfx_xcrun_EXECUTABLE}" --sdk macosx --show-sdk-path
                    OUTPUT_VARIABLE _gfx_llvm_osx_sysroot
                    OUTPUT_STRIP_TRAILING_WHITESPACE
                    ERROR_QUIET)
            endif()
        endif()
        if(_gfx_llvm_osx_sysroot AND EXISTS "${_gfx_llvm_osx_sysroot}")
            list(APPEND _gfx_llvm_toolchain_args
                -DCMAKE_OSX_SYSROOT=${_gfx_llvm_osx_sysroot})
        endif()
    elseif(CMAKE_CROSSCOMPILING)
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$")
            set(_gfx_llvm_targets_to_build "AArch64")
        elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm|armv7|armv7l|ARM)$")
            set(_gfx_llvm_targets_to_build "ARM")
        elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
            set(_gfx_llvm_targets_to_build "X86")
        endif()

        foreach(_gfx_cross_var
                CMAKE_TOOLCHAIN_FILE
                CMAKE_SYSTEM_NAME
                CMAKE_SYSTEM_PROCESSOR
                CMAKE_SYSROOT
                CMAKE_C_COMPILER
                CMAKE_CXX_COMPILER
                CMAKE_C_COMPILER_TARGET
                CMAKE_CXX_COMPILER_TARGET
                CMAKE_AR
                CMAKE_RANLIB
                CMAKE_STRIP
                PKG_CONFIG_EXECUTABLE
                CMAKE_FIND_ROOT_PATH_MODE_PROGRAM
                CMAKE_FIND_ROOT_PATH_MODE_LIBRARY
                CMAKE_FIND_ROOT_PATH_MODE_INCLUDE
                CMAKE_FIND_ROOT_PATH_MODE_PACKAGE)
            if(DEFINED ${_gfx_cross_var} AND NOT "${${_gfx_cross_var}}" STREQUAL "")
                list(APPEND _gfx_llvm_toolchain_args
                    -D${_gfx_cross_var}=${${_gfx_cross_var}})
            endif()
        endforeach()

        foreach(_gfx_cross_list_var
                CMAKE_FIND_ROOT_PATH
                CMAKE_PREFIX_PATH)
            if(DEFINED ${_gfx_cross_list_var} AND NOT "${${_gfx_cross_list_var}}" STREQUAL "")
                list(JOIN ${_gfx_cross_list_var} ";" _gfx_cross_list_value)
                list(APPEND _gfx_llvm_toolchain_args
                    "-D${_gfx_cross_list_var}=${_gfx_cross_list_value}")
            endif()
        endforeach()
    endif()
    if(NOT DEFINED _gfx_llvm_targets_to_build)
        set(_gfx_llvm_targets_to_build "Native")
    endif()

    foreach(_gfx_llvm_flag_var
            CMAKE_C_FLAGS
            CMAKE_CXX_FLAGS
            CMAKE_EXE_LINKER_FLAGS
            CMAKE_SHARED_LINKER_FLAGS
            CMAKE_MODULE_LINKER_FLAGS)
        if(DEFINED ${_gfx_llvm_flag_var} AND NOT "${${_gfx_llvm_flag_var}}" STREQUAL "")
            list(APPEND _gfx_llvm_toolchain_args
                "-D${_gfx_llvm_flag_var}=${${_gfx_llvm_flag_var}}")
        endif()
    endforeach()

    # llvmorg-22.1.2 unconditionally enters mlir-parser-fuzzer from mlir/tools,
    # but the fuzzer subdirs assume add_llvm_fuzzer created a target. Supplying
    # a tiny local fuzzing engine archive keeps llvm-project pristine while
    # allowing the external MLIR build to configure cleanly.
    set(_gfx_llvm_fuzzing_engine_dir "${CMAKE_CURRENT_BINARY_DIR}/gfx_llvm_support")
    file(MAKE_DIRECTORY "${_gfx_llvm_fuzzing_engine_dir}")
    add_library(gfx_llvm_fuzzing_engine_stub STATIC
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/gfx_dummy_fuzzing_engine.c)
    set_target_properties(gfx_llvm_fuzzing_engine_stub PROPERTIES
        ARCHIVE_OUTPUT_DIRECTORY "${_gfx_llvm_fuzzing_engine_dir}"
        OUTPUT_NAME "gfx_dummy_fuzzing_engine")
    set(_gfx_llvm_fuzzing_engine_archive
        "${_gfx_llvm_fuzzing_engine_dir}/${CMAKE_STATIC_LIBRARY_PREFIX}gfx_dummy_fuzzing_engine${CMAKE_STATIC_LIBRARY_SUFFIX}")

    set(_gfx_llvm_byproducts
        "${GFX_LLVM_INSTALL_DIR}/lib/cmake/llvm/LLVMConfig.cmake"
        "${GFX_LLVM_INSTALL_DIR}/lib/cmake/mlir/MLIRConfig.cmake"
        "${GFX_LLVM_BUILD_DIR}/lib/cmake/llvm/LLVMConfig.cmake"
        "${GFX_LLVM_BUILD_DIR}/lib/cmake/mlir/MLIRConfig.cmake")
    foreach(_gfx_llvm_component IN LISTS GFX_LLVM_COMPONENT_LIBS)
        list(APPEND _gfx_llvm_byproducts
            "${GFX_LLVM_INSTALL_DIR}/lib/lib${_gfx_llvm_component}.a"
            "${GFX_LLVM_BUILD_DIR}/lib/lib${_gfx_llvm_component}.a")
    endforeach()
    foreach(_gfx_mlir_component IN LISTS GFX_MLIR_COMPONENT_LIBS)
        list(APPEND _gfx_llvm_byproducts
            "${GFX_LLVM_INSTALL_DIR}/lib/lib${_gfx_mlir_component}.a"
            "${GFX_LLVM_BUILD_DIR}/lib/lib${_gfx_mlir_component}.a")
    endforeach()

    ExternalProject_Add(gfx_llvm_external
        SOURCE_DIR ${GFX_LLVM_PROJECT_SOURCE_DIR}/llvm
        BINARY_DIR ${GFX_LLVM_BUILD_DIR}
        DEPENDS gfx_llvm_fuzzing_engine_stub
        BUILD_BYPRODUCTS
            ${_gfx_llvm_byproducts}
        CMAKE_ARGS
            -G ${CMAKE_GENERATOR}
            -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
            -DCMAKE_OSX_ARCHITECTURES=${CMAKE_OSX_ARCHITECTURES}
            -DCMAKE_INSTALL_PREFIX=${GFX_LLVM_INSTALL_DIR}
            -DLLVM_ENABLE_PROJECTS=mlir
            -DLLVM_TARGETS_TO_BUILD=${_gfx_llvm_targets_to_build}
            -DLLVM_LIB_FUZZING_ENGINE=${_gfx_llvm_fuzzing_engine_archive}
            -DBUILD_SHARED_LIBS=OFF
            -DLLVM_BUILD_LLVM_DYLIB=OFF
            -DLLVM_BUILD_TOOLS=OFF
            -DLLVM_BUILD_UTILS=OFF
            -DLLVM_TOOL_LTO_BUILD=OFF
            -DLLVM_TOOL_LLVM_AS_FUZZER_BUILD=OFF
            -DLLVM_TOOL_LLVM_DIS_FUZZER_BUILD=OFF
            -DLLVM_TOOL_LLVM_DLANG_DEMANGLE_FUZZER_BUILD=OFF
            -DLLVM_TOOL_LLVM_DWARFDUMP_BUILD=OFF
            -DLLVM_TOOL_LLVM_ISEL_FUZZER_BUILD=OFF
            -DLLVM_TOOL_LLVM_ITANIUM_DEMANGLE_FUZZER_BUILD=OFF
            -DLLVM_TOOL_LLVM_MC_ASSEMBLE_FUZZER_BUILD=OFF
            -DLLVM_TOOL_LLVM_MC_DISASSEMBLE_FUZZER_BUILD=OFF
            -DLLVM_TOOL_LLVM_MICROSOFT_DEMANGLE_FUZZER_BUILD=OFF
            -DLLVM_TOOL_LLVM_OPT_FUZZER_BUILD=OFF
            -DLLVM_TOOL_LLVM_RUST_DEMANGLE_FUZZER_BUILD=OFF
            -DLLVM_TOOL_LLVM_SPECIAL_CASE_LIST_FUZZER_BUILD=OFF
            -DLLVM_TOOL_LLVM_YAML_NUMERIC_PARSER_FUZZER_BUILD=OFF
            -DLLVM_TOOL_LLVM_YAML_PARSER_FUZZER_BUILD=OFF
            -DLLVM_TOOL_VFABI_DEMANGLE_FUZZER_BUILD=OFF
            -DLLVM_ENABLE_BINDINGS=OFF
            -DLLVM_INCLUDE_BENCHMARKS=OFF
            -DLLVM_INCLUDE_DOCS=OFF
            -DLLVM_INCLUDE_EXAMPLES=OFF
            -DLLVM_INCLUDE_TESTS=OFF
            -DLLVM_INCLUDE_TOOLS=ON
            -DLLVM_INCLUDE_UTILS=OFF
            -DLLVM_ENABLE_ZLIB=OFF
            -DLLVM_ENABLE_ZSTD=OFF
            -DLLVM_ENABLE_LIBEDIT=OFF
            -DLLVM_ENABLE_LIBXML2=OFF
            -DLLVM_ENABLE_TERMINFO=OFF
            -DLLVM_ENABLE_BACKTRACES=OFF
            -DMLIR_BUILD_TOOLS=OFF
            -DMLIR_INCLUDE_TOOLS=OFF
            -DMLIR_INCLUDE_TESTS=OFF
            -DMLIR_INSTALL_AGGREGATE_OBJECTS=OFF
            -DMLIR_ENABLE_BINDINGS_PYTHON=OFF
            -DLLVM_ENABLE_RTTI=ON
            ${_gfx_llvm_toolchain_args}
        INSTALL_COMMAND ${CMAKE_COMMAND} --build . --target install)

    # Prefer installed package exports once available. They are the stable consumption surface
    # for both host and Android builds; fall back to build-tree exports only before install ran.
    set(_gfx_llvm_build_cmake_dir "${GFX_LLVM_BUILD_DIR}/lib/cmake/llvm")
    set(_gfx_mlir_build_cmake_dir "${GFX_LLVM_BUILD_DIR}/lib/cmake/mlir")
    set(_gfx_llvm_config_dir "${GFX_LLVM_INSTALL_DIR}/lib/cmake/llvm")
    set(_gfx_mlir_config_dir "${GFX_LLVM_INSTALL_DIR}/lib/cmake/mlir")
    if((NOT EXISTS "${_gfx_llvm_config_dir}/LLVMConfig.cmake" OR NOT EXISTS "${_gfx_mlir_config_dir}/MLIRConfig.cmake") AND
       EXISTS "${_gfx_llvm_build_cmake_dir}/LLVMConfig.cmake" AND EXISTS "${_gfx_mlir_build_cmake_dir}/MLIRConfig.cmake")
        set(_gfx_llvm_config_dir "${_gfx_llvm_build_cmake_dir}")
        set(_gfx_mlir_config_dir "${_gfx_mlir_build_cmake_dir}")
    endif()

    set(MLIR_DIR "${_gfx_mlir_config_dir}" CACHE PATH "MLIR config dir for GFX" FORCE)
    set(LLVM_DIR "${_gfx_llvm_config_dir}" CACHE PATH "LLVM config dir for GFX" FORCE)

    if(EXISTS "${LLVM_DIR}/LLVMConfig.cmake" AND EXISTS "${MLIR_DIR}/MLIRConfig.cmake")
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

        foreach(_gfx_llvm_component IN LISTS GFX_LLVM_COMPONENT_LIBS)
            _gfx_import_lib(${_gfx_llvm_component})
        endforeach()
        foreach(_gfx_mlir_component IN LISTS GFX_MLIR_COMPONENT_LIBS)
            _gfx_import_lib(${_gfx_mlir_component})
        endforeach()
        unset(_gfx_llvm_component)
        unset(_gfx_mlir_component)
    endif()

    unset(_gfx_llvm_build_cmake_dir)
    unset(_gfx_mlir_build_cmake_dir)
    unset(_gfx_llvm_config_dir)
    unset(_gfx_mlir_config_dir)
    unset(GFX_LLVM_BUILD_DIR)

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

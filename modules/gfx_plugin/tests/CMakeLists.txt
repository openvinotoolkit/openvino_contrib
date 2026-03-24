set(TARGET_NAME ov_gfx_func_tests)

set(GFX_TEST_DEPENDENCIES openvino_gfx_plugin)
set(GFX_TEST_LINK_LIBS
    openvino::funcSharedTests
    openvino::runtime::dev
    openvino_gfx_plugin
    gmock
    gtest_main
    gtest
    gfx_runtime_common
    gfx_runtime_mlir
    gfx_mlir
    gfx_plugin_core
    MLIRInferIntRangeCommon
    MLIRInferIntRangeInterface
    MLIRTransformUtils
    MLIRAffineUtils
    MLIRAffineDialect
    MLIRAffineTransforms
    MLIRAffineAnalysis
    MLIRIR
    MLIRSupport
    MLIRCastInterfaces
    MLIRReconcileUnrealizedCasts
    MLIRTensorDialect
    MLIRArithDialect
    MLIRSCFDialect
    MLIRMathDialect
    MLIRComplexDialect
    MLIRLinalgDialect
    MLIRLinalgTransforms
    MLIRLinalgUtils
    MLIRFuncDialect
    MLIRFuncTransforms
    MLIRBufferizationTransforms
    MLIRParser
    MLIRInferTypeOpInterface
    MLIRValueBoundsOpInterface
    MLIRAnalysis
    MLIRDialectUtils)

if(NOT APPLE)
    list(APPEND GFX_TEST_LINK_LIBS gfx_runtime_vulkan)
endif()

# Align with CPU functional tests: single target built via ov_add_test_target,
# no custom archives or force-load.
set(GFX_TEST_EXCLUDED_PATHS "")
list(APPEND GFX_TEST_EXCLUDED_PATHS
    ${CMAKE_CURRENT_SOURCE_DIR}/tools
)
if(NOT GFX_BACKEND_VULKAN_AVAILABLE)
    list(APPEND GFX_TEST_EXCLUDED_PATHS
        ${CMAKE_CURRENT_SOURCE_DIR}/backends/vulkan
        ${CMAKE_CURRENT_SOURCE_DIR}/backends/vulkan/vulkan_runtime_test.cpp
    )
endif()
if(APPLE)
    list(APPEND GFX_TEST_EXCLUDED_PATHS
        ${CMAKE_CURRENT_SOURCE_DIR}/backends/metal/memory_test_stub.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/backends/metal/set_device_name_stub.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/backends/vulkan/gpu_backend_test_stub.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/backends/vulkan/gpu_stage_activation_test_stub.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/backends/vulkan/gpu_stage_factory_test_stub.cpp
    )
else()
    list(APPEND GFX_TEST_EXCLUDED_PATHS
        ${CMAKE_CURRENT_SOURCE_DIR}/backends/metal/memory_test.mm
        ${CMAKE_CURRENT_SOURCE_DIR}/backends/metal/gpu_backend_test.mm
        ${CMAKE_CURRENT_SOURCE_DIR}/backends/metal/gpu_stage_activation_test.mm
        ${CMAKE_CURRENT_SOURCE_DIR}/backends/metal/gpu_stage_factory_test.mm
        ${CMAKE_CURRENT_SOURCE_DIR}/backends/metal/gpu_memory_copy_test.mm
        ${CMAKE_CURRENT_SOURCE_DIR}/backends/metal/set_device_name.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/backends/metal/set_device_name_stub.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/backends/vulkan/vulkan_runtime_test_stub.cpp
    )
endif()

ov_add_test_target(
    NAME ${TARGET_NAME}
    ROOT ${CMAKE_CURRENT_SOURCE_DIR}
    EXCLUDED_SOURCE_PATHS ${GFX_TEST_EXCLUDED_PATHS}
    DEPENDENCIES ${GFX_TEST_DEPENDENCIES}
    LINK_LIBRARIES ${GFX_TEST_LINK_LIBS}
    LABELS
        OV GFX
)

# Add ObjC++ runtime tests for memory manager.
target_sources(${TARGET_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/gpu_stage_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/gfx_parallelism_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/layout_cleanup_test.cpp
)

if(APPLE)
    target_sources(${TARGET_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/backends/metal/memory_test.mm
        ${CMAKE_CURRENT_SOURCE_DIR}/backends/metal/gpu_memory_copy_test.mm
        ${CMAKE_CURRENT_SOURCE_DIR}/backends/metal/gpu_stage_factory_test.mm
        ${CMAKE_CURRENT_SOURCE_DIR}/backends/metal/gpu_stage_activation_test.mm
        ${CMAKE_CURRENT_SOURCE_DIR}/backends/metal/gpu_backend_test.mm
        ${CMAKE_CURRENT_SOURCE_DIR}/backends/vulkan/vulkan_runtime_test_stub.cpp
    )
    target_link_libraries(${TARGET_NAME} PRIVATE gfx_runtime_metal)
else()
    target_sources(${TARGET_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/backends/metal/runtime/dtype.cpp
    )
endif()

if(GFX_BACKEND_VULKAN_AVAILABLE AND TARGET gfx_runtime_vulkan)
    target_sources(${TARGET_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/backends/vulkan/vulkan_runtime_test.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/backends/vulkan/gpu_memory_copy_test.cpp
    )
    if(NOT APPLE)
        target_sources(${TARGET_NAME} PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/backends/vulkan/set_device_name_stub.cpp
        )
    endif()
endif()

# Device-specific bits: path to the built plugin and access to local src headers.
target_compile_definitions(${TARGET_NAME} PRIVATE
    GFX_PLUGIN_PATH="$<TARGET_FILE:openvino_gfx_plugin>"
    GFX_BACKEND_METAL_AVAILABLE=$<BOOL:${GFX_BACKEND_METAL_AVAILABLE}>
    GFX_BACKEND_VULKAN_AVAILABLE=$<BOOL:${GFX_BACKEND_VULKAN_AVAILABLE}>)
target_include_directories(${TARGET_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../src
    ${CMAKE_CURRENT_BINARY_DIR}/../src
    ${OpenVINO_SOURCE_DIR}/src
    ${OpenVINO_SOURCE_DIR}/src/plugins/template/include
)

if(TARGET gfx_plugin_build_settings)
    target_link_libraries(${TARGET_NAME} PRIVATE gfx_plugin_build_settings)
endif()

target_include_directories(${TARGET_NAME} PRIVATE ${MLIR_INCLUDE_DIRS})

if(TARGET GFX::Metal)
    target_link_libraries(${TARGET_NAME} PRIVATE GFX::Metal)
endif()
if(TARGET GFX::Vulkan)
    target_link_libraries(${TARGET_NAME} PRIVATE GFX::Vulkan)
endif()

add_executable(ov_gfx_compare_runner
    ${CMAKE_CURRENT_SOURCE_DIR}/tools/ov_gfx_compare_runner.cpp
)
target_link_libraries(ov_gfx_compare_runner PRIVATE
    openvino::runtime
    openvino_gfx_plugin
    common_test_utils
)
if(TARGET gfx_plugin_build_settings)
    target_link_libraries(ov_gfx_compare_runner PRIVATE gfx_plugin_build_settings)
endif()
target_compile_definitions(ov_gfx_compare_runner PRIVATE
    GFX_PLUGIN_PATH="$<TARGET_FILE:openvino_gfx_plugin>"
)

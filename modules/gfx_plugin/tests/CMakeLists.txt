set(TARGET_NAME ov_gfx_func_tests)

set(GFX_TEST_DEPENDENCIES openvino_gfx_plugin)
set(GFX_TEST_LINK_LIBS
    openvino::funcSharedTests
    openvino::runtime::dev
    openvino_gfx_plugin
    gmock
    gtest_main
    gtest)

# Align with CPU functional tests: single target built via ov_add_test_target,
# no custom archives or force-load.
set(GFX_TEST_EXCLUDED_PATHS "")
if(NOT GFX_BACKEND_VULKAN_AVAILABLE)
    list(APPEND GFX_TEST_EXCLUDED_PATHS
        ${CMAKE_CURRENT_SOURCE_DIR}/backends/vulkan
        ${CMAKE_CURRENT_SOURCE_DIR}/backends/vulkan/vulkan_runtime_test.cpp
    )
endif()
if(APPLE)
    list(APPEND GFX_TEST_EXCLUDED_PATHS
        ${CMAKE_CURRENT_SOURCE_DIR}/unit/set_device_name.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/backends/metal/memory_test_stub.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/unit/gpu_backend_test_stub.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/unit/gpu_stage_activation_test_stub.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/unit/gpu_stage_factory_test_stub.cpp
    )
else()
    list(APPEND GFX_TEST_EXCLUDED_PATHS
        ${CMAKE_CURRENT_SOURCE_DIR}/backends/metal/memory_test.mm
        ${CMAKE_CURRENT_SOURCE_DIR}/unit/gpu_backend_test.mm
        ${CMAKE_CURRENT_SOURCE_DIR}/unit/gpu_stage_activation_test.mm
        ${CMAKE_CURRENT_SOURCE_DIR}/unit/gpu_stage_factory_test.mm
        ${CMAKE_CURRENT_SOURCE_DIR}/backends/metal/set_device_name.cpp
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
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/gpu_memory_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/transforms/pipeline.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/transforms/conv_relu_fusion.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/runtime/memory_manager.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/runtime/gfx_backend_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/kernel_ir/gfx_kernel_cache.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/runtime/gfx_logger.cpp
)
if(NOT GFX_BACKEND_VULKAN_AVAILABLE)
    target_sources(${TARGET_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/backends/vulkan/runtime/memory_api_stub.cpp
    )
endif()
if(NOT APPLE)
    target_sources(${TARGET_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/backends/metal/runtime/runtime_stub.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/backends/metal/runtime/dtype.cpp
    )
endif()

if(APPLE)
    target_sources(${TARGET_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/backends/metal/memory_test.mm
        ${CMAKE_CURRENT_SOURCE_DIR}/unit/gpu_stage_factory_test.mm
        ${CMAKE_CURRENT_SOURCE_DIR}/unit/gpu_stage_activation_test.mm
        ${CMAKE_CURRENT_SOURCE_DIR}/unit/gpu_backend_test.mm
        ${CMAKE_CURRENT_SOURCE_DIR}/backends/vulkan/vulkan_runtime_test_stub.cpp
    )
endif()

if(TARGET gfx_runtime_metal_obj)
    target_sources(${TARGET_NAME} PRIVATE $<TARGET_OBJECTS:gfx_runtime_metal_obj>)
endif()
if(GFX_BACKEND_VULKAN_AVAILABLE AND TARGET gfx_runtime_vulkan_obj)
    target_sources(${TARGET_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/backends/vulkan/vulkan_runtime_test.cpp
        $<TARGET_OBJECTS:gfx_runtime_vulkan_obj>)
endif()

# Device-specific bits: path to the built plugin and access to local src headers.
target_compile_definitions(${TARGET_NAME} PRIVATE
    GFX_PLUGIN_PATH="$<TARGET_FILE:openvino_gfx_plugin>"
    GFX_BACKEND_METAL_AVAILABLE=$<BOOL:${GFX_BACKEND_METAL_AVAILABLE}>
    GFX_BACKEND_VULKAN_AVAILABLE=$<BOOL:${GFX_BACKEND_VULKAN_AVAILABLE}>)
target_include_directories(${TARGET_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../src
    ${CMAKE_CURRENT_BINARY_DIR}/../src
    ${OpenVINO_SOURCE_DIR}/src
    ${OpenVINO_SOURCE_DIR}/src/plugins/template/include
)

if(TARGET gfx_plugin_build_settings)
    target_link_libraries(${TARGET_NAME} PRIVATE gfx_plugin_build_settings)
endif()

if(ENABLE_GFX_MLIR)
    target_include_directories(${TARGET_NAME} PRIVATE ${MLIR_INCLUDE_DIRS})
    target_link_libraries(${TARGET_NAME} PRIVATE gfx_mlir gfx_mlir_codegen)
endif()

if(TARGET GFX::Metal)
    target_link_libraries(${TARGET_NAME} PRIVATE GFX::Metal)
endif()
if(TARGET GFX::Vulkan)
    target_link_libraries(${TARGET_NAME} PRIVATE GFX::Vulkan)
endif()
if(TARGET gfx_vulkan_mlir_deps)
    target_link_libraries(${TARGET_NAME} PRIVATE gfx_vulkan_mlir_deps)
endif()

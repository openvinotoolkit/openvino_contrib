set(TARGET_NAME ov_metal_func_tests)

set(METAL_TEST_DEPENDENCIES openvino_metal_plugin)
set(METAL_TEST_LINK_LIBS
    openvino::funcSharedTests
    openvino::runtime::dev
    openvino_metal_plugin
    gmock
    gtest_main
    gtest)

# Template plugin may be absent in the Developer Package; import it if available.
if(NOT TARGET openvino_template_plugin)
    find_library(TEMPLATE_PLUGIN_LIB
        NAMES openvino_template_plugin
        PATHS
            ${OpenVINODeveloperPackage_DIR}/bin
            ${OpenVINODeveloperPackage_DIR}/lib
            ${OpenVINODeveloperPackage_DIR}/share
        NO_DEFAULT_PATH)
    if(TEMPLATE_PLUGIN_LIB)
        add_library(openvino_template_plugin SHARED IMPORTED)
        set_target_properties(openvino_template_plugin PROPERTIES
            IMPORTED_LOCATION "${TEMPLATE_PLUGIN_LIB}")
    endif()
endif()

if(TARGET openvino_template_plugin)
    list(APPEND METAL_TEST_DEPENDENCIES openvino_template_plugin)
    list(APPEND METAL_TEST_LINK_LIBS openvino_template_plugin)
endif()

# Align with CPU functional tests: single target built via ov_add_test_target,
# no custom archives or force-load.
ov_add_test_target(
    NAME ${TARGET_NAME}
    ROOT ${CMAKE_CURRENT_SOURCE_DIR}
    DEPENDENCIES ${METAL_TEST_DEPENDENCIES}
    LINK_LIBRARIES ${METAL_TEST_LINK_LIBS}
    LABELS
        OV METAL
)

# Add ObjC++ runtime tests for memory manager.
target_sources(${TARGET_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/metal_memory_test.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/metal_op_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/metal_op_factory_test.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/metal_activation_op_test.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/runtime/metal_memory.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/runtime/memory/metal_device_caps.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/runtime/memory/metal_allocator_core.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/runtime/memory/metal_heap_pool.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/runtime/memory/metal_staging_pool.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/runtime/memory/metal_allocator.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/runtime/memory/metal_memory_session.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/runtime/memory/metal_const_cache.mm
)

# Device-specific bits: path to the built plugin and access to local src headers.
target_compile_definitions(${TARGET_NAME} PRIVATE METAL_PLUGIN_PATH="$<TARGET_FILE:openvino_metal_plugin>")
if(TARGET openvino_template_plugin)
    target_compile_definitions(${TARGET_NAME} PRIVATE TEMPLATE_PLUGIN_PATH="$<TARGET_FILE:openvino_template_plugin>")
endif()
target_include_directories(${TARGET_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../src
    ${OpenVINO_SOURCE_DIR}/src
    ${OpenVINO_SOURCE_DIR}/src/plugins/template/include
)

if(ENABLE_METAL_MLIR)
    target_include_directories(${TARGET_NAME} PRIVATE ${MLIR_INCLUDE_DIRS})
endif()

if(APPLE)
    target_link_libraries(${TARGET_NAME} PRIVATE "-framework Metal" "-framework Foundation")
endif()

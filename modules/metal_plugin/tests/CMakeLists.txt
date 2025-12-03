set(TARGET_NAME ov_metal_basic_tests)

# Align with CPU functional tests: single target built via ov_add_test_target,
# no custom archives or force-load.
ov_add_test_target(
    NAME ${TARGET_NAME}
    ROOT ${CMAKE_CURRENT_SOURCE_DIR}
    DEPENDENCIES
        openvino_metal_plugin
    LINK_LIBRARIES
        openvino::funcSharedTests
        openvino::runtime::dev
        gmock
        gtest_main
        gtest
    LABELS
        OV METAL
)

# Device-specific bits: path to the built plugin and access to local src headers.
target_compile_definitions(${TARGET_NAME} PRIVATE METAL_PLUGIN_PATH="$<TARGET_FILE:openvino_metal_plugin>")
target_include_directories(${TARGET_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)

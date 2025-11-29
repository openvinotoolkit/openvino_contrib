set(TARGET_NAME ov_metal_basic_tests)

set(GTEST_ROOT "/Users/anesterov/repos/openvino_contrib/cmake-build-artifacts/openvino/thirdparty/gtest/gtest/googletest")
add_library(ov_metal_gtest STATIC
    ${GTEST_ROOT}/src/gtest-all.cc
    ${GTEST_ROOT}/src/gtest_main.cc)
target_include_directories(ov_metal_gtest PUBLIC
    ${GTEST_ROOT}/include
    ${GTEST_ROOT})

ov_add_test_target(
    NAME ${TARGET_NAME}
    ROOT ${CMAKE_CURRENT_SOURCE_DIR}
    DEPENDENCIES
        openvino_metal_plugin
    LINK_LIBRARIES
        openvino::runtime::dev
        ov_metal_gtest
    ADD_CLANG_FORMAT
    LABELS
        OV UNIT METAL
)

target_sources(${TARGET_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/metal_basic_ops_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/metal_precision_study_test.cpp)

target_compile_definitions(${TARGET_NAME} PRIVATE METAL_PLUGIN_PATH="$<TARGET_FILE:openvino_metal_plugin>")

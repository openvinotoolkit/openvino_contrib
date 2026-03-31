set(TARGET_NAME ov_gfx_func_tests)
set(UNIT_TARGET_NAME ov_gfx_unit_tests)
set(RUNTIME_MICRO_TARGET_NAME ov_gfx_runtime_micro_tests)

set(GFX_TEST_DEPENDENCIES openvino_gfx_plugin)
set(GFX_TEST_LINK_LIBS
    openvino::funcSharedTests
    openvino::runtime::dev
    openvino::runtime
    gmock
    gtest_main
    gtest
    gfx_runtime_common)

set(GFX_UNIT_TEST_LINK_LIBS
    openvino::funcSharedTests
    openvino::runtime::dev
    openvino::runtime
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
    list(APPEND GFX_UNIT_TEST_LINK_LIBS gfx_runtime_vulkan)
endif()

# Align with CPU functional tests: single target built via ov_add_test_target,
# no custom archives or force-load.
set(GFX_TEST_EXCLUDED_PATHS "")
list(APPEND GFX_TEST_EXCLUDED_PATHS
    ${CMAKE_CURRENT_SOURCE_DIR}/tools
    ${CMAKE_CURRENT_SOURCE_DIR}/unit
    ${CMAKE_CURRENT_SOURCE_DIR}/backends/metal/memory_test.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/backends/metal/memory_test_stub.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/backends/metal/gpu_backend_test.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/backends/metal/gpu_memory_copy_test.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/backends/metal/gpu_stage_activation_test.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/backends/metal/gpu_stage_factory_test.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/backends/vulkan/gpu_backend_test_stub.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/backends/vulkan/gpu_memory_copy_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/backends/vulkan/gpu_stage_activation_test_stub.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/backends/vulkan/gpu_stage_factory_test_stub.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/backends/vulkan/vulkan_runtime_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/backends/vulkan/vulkan_runtime_test_stub.cpp
)
if(NOT GFX_BACKEND_VULKAN_AVAILABLE)
    list(APPEND GFX_TEST_EXCLUDED_PATHS
        ${CMAKE_CURRENT_SOURCE_DIR}/backends/vulkan
    )
endif()
if(APPLE)
    list(APPEND GFX_TEST_EXCLUDED_PATHS
        ${CMAKE_CURRENT_SOURCE_DIR}/backends/metal/set_device_name_stub.cpp
    )
else()
    list(APPEND GFX_TEST_EXCLUDED_PATHS
        ${CMAKE_CURRENT_SOURCE_DIR}/backends/metal/basic_ops_test.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/backends/metal/set_device_name.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/backends/metal/set_device_name_stub.cpp
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

target_sources(${TARGET_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/plugin_tests.cpp
)
if(APPLE)
    target_sources(${TARGET_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/unit/memory_device_integration_test.mm
    )
endif()

# Device-specific bits: path to the built plugin and access to local src headers.
target_compile_definitions(${TARGET_NAME} PRIVATE
    GFX_PLUGIN_PATH="$<TARGET_FILE:openvino_gfx_plugin>"
    GFX_BACKEND_METAL_AVAILABLE=$<BOOL:${GFX_BACKEND_METAL_AVAILABLE}>
    GFX_BACKEND_VULKAN_AVAILABLE=$<BOOL:${GFX_BACKEND_VULKAN_AVAILABLE}>)
target_include_directories(${TARGET_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../src
    ${CMAKE_CURRENT_BINARY_DIR}/../src
    ${OpenVINO_SOURCE_DIR}/src
    ${OpenVINO_SOURCE_DIR}/src/plugins/template/include
)

if(TARGET gfx_plugin_build_settings)
    target_link_libraries(${TARGET_NAME} PRIVATE gfx_plugin_build_settings)
endif()
if(APPLE)
    set_tests_properties(${TARGET_NAME} PROPERTIES ENVIRONMENT "DISABLE_PLUGIN_CACHE=1")
endif()

if(TARGET GFX::Metal)
    target_link_libraries(${TARGET_NAME} PRIVATE GFX::Metal)
endif()
if(TARGET GFX::Vulkan)
    target_link_libraries(${TARGET_NAME} PRIVATE GFX::Vulkan)
endif()

add_executable(${UNIT_TARGET_NAME})
add_dependencies(${UNIT_TARGET_NAME} ${GFX_TEST_DEPENDENCIES})
target_sources(${UNIT_TARGET_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/gpu_backend_base_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/gpu_const_cache_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/kernel_arg_reuse_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/infer_submission_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/infer_pipeline_reuse_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/gfx_profiling_report_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/ov_gfx_microbench_calibration_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/gpu_stage_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/gfx_stage_policy_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/gfx_parallelism_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/basic_ops_internal_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/layout_cleanup_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/mlir_matmul_parallel_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/mlir_conv3d_lowering_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/mlir_conv_parallel_test.cpp)
if(APPLE)
    target_sources(${UNIT_TARGET_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/backends/metal/set_device_name.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/backends/metal/memory_test.mm
        ${CMAKE_CURRENT_SOURCE_DIR}/backends/metal/gpu_memory_copy_test.mm
        ${CMAKE_CURRENT_SOURCE_DIR}/backends/metal/gpu_stage_factory_test.mm
        ${CMAKE_CURRENT_SOURCE_DIR}/backends/metal/gpu_stage_activation_test.mm
        ${CMAKE_CURRENT_SOURCE_DIR}/backends/metal/gpu_backend_test.mm
        ${CMAKE_CURRENT_SOURCE_DIR}/backends/vulkan/vulkan_runtime_test_stub.cpp)
else()
    target_sources(${UNIT_TARGET_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/backends/metal/memory_test_stub.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/backends/vulkan/gpu_backend_test_stub.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/backends/vulkan/gpu_stage_activation_test_stub.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/backends/vulkan/gpu_stage_factory_test_stub.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/backends/vulkan/gpu_memory_copy_test.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/backends/vulkan/vulkan_runtime_test.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/backends/vulkan/set_device_name_stub.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/backends/metal/runtime/dtype.cpp)
endif()
target_link_libraries(${UNIT_TARGET_NAME} PRIVATE ${GFX_UNIT_TEST_LINK_LIBS})
target_compile_definitions(${UNIT_TARGET_NAME} PRIVATE
    GFX_PLUGIN_PATH="$<TARGET_FILE:openvino_gfx_plugin>"
    GFX_BACKEND_METAL_AVAILABLE=$<BOOL:${GFX_BACKEND_METAL_AVAILABLE}>
    GFX_BACKEND_VULKAN_AVAILABLE=$<BOOL:${GFX_BACKEND_VULKAN_AVAILABLE}>)
target_include_directories(${UNIT_TARGET_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../src
    ${CMAKE_CURRENT_BINARY_DIR}/../src
    ${OpenVINO_SOURCE_DIR}/src
    ${OpenVINO_SOURCE_DIR}/src/plugins/template/include
    ${OpenVINO_SOURCE_DIR}/src/tests/test_utils/common_test_utils/include
    ${OpenVINO_SOURCE_DIR}/src/tests/test_utils/functional_test_utils/include
    ${OpenVINO_SOURCE_DIR}/src/tests/functional/plugin/shared/include
    ${OpenVINO_SOURCE_DIR}/src/tests/functional/base_func_tests/include
    ${OpenVINO_SOURCE_DIR}/src/tests/ov_helpers/ov_lpt_models/include
    ${MLIR_INCLUDE_DIRS})
if(TARGET gfx_plugin_build_settings)
    target_link_libraries(${UNIT_TARGET_NAME} PRIVATE gfx_plugin_build_settings)
endif()
if(APPLE)
    target_link_libraries(${UNIT_TARGET_NAME} PRIVATE gfx_runtime_metal)
endif()
if(TARGET GFX::Metal)
    target_link_libraries(${UNIT_TARGET_NAME} PRIVATE GFX::Metal)
endif()
if(TARGET GFX::Vulkan)
    target_link_libraries(${UNIT_TARGET_NAME} PRIVATE GFX::Vulkan)
endif()
add_test(NAME ${UNIT_TARGET_NAME} COMMAND ${UNIT_TARGET_NAME})
set_tests_properties(${UNIT_TARGET_NAME} PROPERTIES LABELS "OV;GFX")

add_executable(${RUNTIME_MICRO_TARGET_NAME})
add_dependencies(${RUNTIME_MICRO_TARGET_NAME} ${GFX_TEST_DEPENDENCIES})
target_sources(${RUNTIME_MICRO_TARGET_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/runtime_subgraph_test.cpp)
target_link_libraries(${RUNTIME_MICRO_TARGET_NAME} PRIVATE
    openvino::runtime::dev
    openvino::runtime
    common_test_utils
    gmock
    gtest_main
    gtest)
target_compile_definitions(${RUNTIME_MICRO_TARGET_NAME} PRIVATE
    GFX_PLUGIN_PATH="$<TARGET_FILE:openvino_gfx_plugin>"
    GFX_BACKEND_METAL_AVAILABLE=$<BOOL:${GFX_BACKEND_METAL_AVAILABLE}>
    GFX_BACKEND_VULKAN_AVAILABLE=$<BOOL:${GFX_BACKEND_VULKAN_AVAILABLE}>)
target_include_directories(${RUNTIME_MICRO_TARGET_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${OpenVINO_SOURCE_DIR}/src
    ${OpenVINO_SOURCE_DIR}/src/tests/test_utils/common_test_utils/include
    ${OpenVINO_SOURCE_DIR}/src/tests/test_utils/functional_test_utils/include
    ${OpenVINO_SOURCE_DIR}/src/tests/functional/plugin/shared/include
    ${OpenVINO_SOURCE_DIR}/src/tests/functional/base_func_tests/include)
if(TARGET gfx_plugin_build_settings)
    target_link_libraries(${RUNTIME_MICRO_TARGET_NAME} PRIVATE gfx_plugin_build_settings)
endif()
add_test(NAME ${RUNTIME_MICRO_TARGET_NAME} COMMAND ${RUNTIME_MICRO_TARGET_NAME})
set_tests_properties(${RUNTIME_MICRO_TARGET_NAME} PROPERTIES LABELS "OV;GFX")

add_executable(ov_gfx_compare_runner
    ${CMAKE_CURRENT_SOURCE_DIR}/tools/ov_gfx_compare_runner.cpp
)
target_link_libraries(ov_gfx_compare_runner PRIVATE
    openvino::runtime
    common_test_utils
)
if(TARGET gfx_plugin_build_settings)
    target_link_libraries(ov_gfx_compare_runner PRIVATE gfx_plugin_build_settings)
endif()
target_compile_definitions(ov_gfx_compare_runner PRIVATE
    GFX_PLUGIN_PATH="$<TARGET_FILE:openvino_gfx_plugin>"
)

add_executable(ov_gfx_microbench
    ${CMAKE_CURRENT_SOURCE_DIR}/tools/ov_gfx_microbench.cpp
)
if(APPLE)
    target_sources(ov_gfx_microbench PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/tools/ov_gfx_microbench_metal.mm
    )
endif()
target_link_libraries(ov_gfx_microbench PRIVATE
    openvino::runtime
    gfx_runtime_common
)
if(APPLE)
    target_link_libraries(ov_gfx_microbench PRIVATE gfx_runtime_metal)
else()
    target_link_libraries(ov_gfx_microbench PRIVATE gfx_runtime_vulkan)
endif()
if(TARGET GFX::Metal)
    target_link_libraries(ov_gfx_microbench PRIVATE GFX::Metal)
endif()
if(TARGET GFX::Vulkan)
    target_link_libraries(ov_gfx_microbench PRIVATE GFX::Vulkan)
endif()
if(TARGET gfx_plugin_build_settings)
    target_link_libraries(ov_gfx_microbench PRIVATE gfx_plugin_build_settings)
endif()
target_compile_definitions(ov_gfx_microbench PRIVATE
    GFX_PLUGIN_PATH="$<TARGET_FILE:openvino_gfx_plugin>"
)
target_include_directories(ov_gfx_microbench PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../src
    ${CMAKE_CURRENT_BINARY_DIR}/../src
    ${OpenVINO_SOURCE_DIR}/src
    ${OpenVINO_SOURCE_DIR}/src/plugins/template/include
)

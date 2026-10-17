set(TARGET_NAME ov_gfx_func_tests)
set(UNIT_TARGET_NAME ov_gfx_unit_tests)
set(RUNTIME_MICRO_TARGET_NAME ov_gfx_runtime_micro_tests)

find_package(Python3 COMPONENTS Interpreter REQUIRED)

set(GFX_GTEST_MATRIX_TOOL
    "${CMAKE_CURRENT_SOURCE_DIR}/tools/gfx_gtest_matrix.py")
set(GFX_GTEST_MATRIX_REFERENCE_ROOTS "" CACHE STRING
    "Semicolon-separated LABEL=DIR roots with captured GFX gtest matrices to compare against this build")

set(GFX_MEMORY_DEVICE_CONTRACT_NATIVE_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/memory_device_integration_test.mm)
set(GFX_MEMORY_DEVICE_CONTRACT_UNAVAILABLE_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/memory_device_integration_unavailable_test.cpp)
set(GFX_OPENCL_DEVICE_PROFILE_CONTRACT_NATIVE_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/gfx_opencl_device_profile_contract_test.cpp)
set(GFX_OPENCL_DEVICE_PROFILE_CONTRACT_UNAVAILABLE_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/gfx_opencl_device_profile_contract_unavailable_test.cpp)
set(GFX_OPENCL_RUNTIME_BUNDLE_CONTRACT_NATIVE_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/gfx_opencl_runtime_bundle_contract_test.cpp)
set(GFX_OPENCL_RUNTIME_BUNDLE_CONTRACT_UNAVAILABLE_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/gfx_opencl_runtime_bundle_contract_unavailable_test.cpp)
set(GFX_OPENCL_REMOTE_TENSOR_CONTRACT_NATIVE_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/gfx_opencl_remote_tensor_contract_test.cpp)
set(GFX_OPENCL_REMOTE_TENSOR_CONTRACT_UNAVAILABLE_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/gfx_opencl_remote_tensor_contract_unavailable_test.cpp)
set(GFX_METAL_NATIVE_CONTRACT_NATIVE_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/backends/metal/memory_test.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/backends/metal/gpu_memory_copy_test.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/backends/metal/gpu_stage_factory_test.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/backends/metal/gpu_stage_activation_test.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/backends/metal/gpu_backend_test.mm)
set(GFX_METAL_NATIVE_CONTRACT_UNAVAILABLE_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/metal_native_contract_unavailable_test.cpp)

set(GFX_TEST_PLUGINS_XML "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/plugins.xml")
add_custom_target(gfx_test_plugins_xml
    COMMAND "${CMAKE_COMMAND}"
        -DOUTPUT_FILE=${GFX_TEST_PLUGINS_XML}
        -P "${CMAKE_CURRENT_SOURCE_DIR}/../cmake/WriteGfxTestPluginsXml.cmake"
    DEPENDS openvino_gfx_plugin
    COMMENT "Write controlled plugins.xml for GFX tests"
    VERBATIM)
if(TARGET openvino_template_plugin)
    add_dependencies(gfx_test_plugins_xml openvino_template_plugin)
endif()
if(TARGET openvino_auto_plugin)
    add_dependencies(gfx_test_plugins_xml openvino_auto_plugin)
endif()

set(GFX_TEST_DEPENDENCIES openvino_gfx_plugin)
if(DEFINED GFX_RASPBERRY_OPENCL_BUNDLE_TARGET AND TARGET ${GFX_RASPBERRY_OPENCL_BUNDLE_TARGET})
    list(APPEND GFX_TEST_DEPENDENCIES ${GFX_RASPBERRY_OPENCL_BUNDLE_TARGET})
endif()
set(GFX_TEST_RUNTIME_ENVIRONMENT
    "GFX_PLUGIN_PATH=$<TARGET_FILE:openvino_gfx_plugin>")
if(APPLE)
    list(APPEND GFX_TEST_RUNTIME_ENVIRONMENT "DISABLE_PLUGIN_CACHE=1")
endif()
set(GFX_TEST_LINK_LIBS
    openvino::funcSharedTests
    openvino::runtime::dev
    openvino::runtime
    gmock
    gtest_main
    gtest
    gfx_runtime_common
    gfx_mlir_stage_support
    gfx_compiler_common)

set(GFX_UNIT_TEST_LINK_LIBS
    openvino::funcSharedTests
    openvino::runtime::dev
    openvino::runtime
    gmock
    gtest_main
    gtest
    common_test_utils
    gfx_runtime_common
    gfx_mlir_stage_support
    gfx_compiler_common
    gfx_metal_mpsrt_contract
    gfx_mlir
    gfx_plugin_core
    gfx_backend_module_provider
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

if(GFX_BACKEND_OPENCL_AVAILABLE)
    list(APPEND GFX_UNIT_TEST_LINK_LIBS gfx_plugin_opencl gfx_runtime_opencl)
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
)
if(APPLE)
    list(APPEND GFX_TEST_EXCLUDED_PATHS
        ${CMAKE_CURRENT_SOURCE_DIR}/backends/metal/set_device_name_stub.cpp
    )
else()
    list(APPEND GFX_TEST_EXCLUDED_PATHS
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
        ${GFX_MEMORY_DEVICE_CONTRACT_NATIVE_SOURCES}
    )
else()
    target_sources(${TARGET_NAME} PRIVATE
        ${GFX_MEMORY_DEVICE_CONTRACT_UNAVAILABLE_SOURCES}
        ${CMAKE_CURRENT_SOURCE_DIR}/backends/metal/set_device_name_stub.cpp
    )
endif()

add_dependencies(${TARGET_NAME} gfx_test_plugins_xml)

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
set_tests_properties(${TARGET_NAME} PROPERTIES ENVIRONMENT "${GFX_TEST_RUNTIME_ENVIRONMENT}")

if(TARGET GFX::Metal)
    target_link_libraries(${TARGET_NAME} PRIVATE GFX::Metal)
endif()

add_executable(${UNIT_TARGET_NAME})
add_dependencies(${UNIT_TARGET_NAME} ${GFX_TEST_DEPENDENCIES})
set(GFX_UNIT_RUNTIME_EXECUTION_POLICY_SOURCE
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/gfx_runtime_execution_fork.cpp)
if(APPLE OR WIN32 OR ANDROID OR GFX_BACKEND_OPENCL_AVAILABLE)
    set(GFX_UNIT_RUNTIME_EXECUTION_POLICY_SOURCE
        ${CMAKE_CURRENT_SOURCE_DIR}/unit/gfx_runtime_execution_inline.cpp)
endif()
target_sources(${UNIT_TARGET_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/gfx_backend_contracts.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/gfx_backend_architecture_contract_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/gfx_apple_vendor_contract_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/gfx_backend_artifact_payload_contract_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/gfx_const_tensor_descriptor_contract_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/gfx_memory_cache_contract_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/gfx_pipeline_stage_materialization_contract_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/gfx_runtime_descriptor_contract_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/gfx_runtime_param_descriptor_contract_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/gfx_backend_module_contract_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/gfx_kernel_registry_contract_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/gfx_cache_public_contract_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/gfx_runtime_model_runner.cpp
    ${GFX_UNIT_RUNTIME_EXECUTION_POLICY_SOURCE}
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/gfx_runtime_execution_plan_contract_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/gfx_activation_kernel_contract_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/gfx_activation_msl_contract_cases.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/gfx_activation_opencl_contract_cases.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/gfx_eltwise_kernel_contract_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/gfx_eltwise_opencl_contract_cases.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/gfx_eltwise_opencl_source_artifacts_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/gfx_reduction_kernel_contract_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/gfx_shapeof_kernel_contract_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/gfx_softmax_kernel_contract_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/gfx_conv_kernel_contract_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/gfx_pool_kernel_contract_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/gfx_interpolate_kernel_contract_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/gfx_matmul_kernel_contract_test.cpp
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
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/gfx_opencl_concat_split_source_artifacts_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/gfx_opencl_gather_scatter_source_artifacts_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/gfx_opencl_layout_source_artifacts_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/gfx_opencl_range_tile_source_artifacts_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/gfx_opencl_source_artifacts_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/layout_cleanup_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/mlir_matmul_parallel_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/mlir_conv3d_lowering_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/mlir_conv_parallel_test.cpp)
if(GFX_BACKEND_OPENCL_AVAILABLE)
    target_sources(${UNIT_TARGET_NAME} PRIVATE
        ${GFX_OPENCL_DEVICE_PROFILE_CONTRACT_NATIVE_SOURCES}
        ${GFX_OPENCL_RUNTIME_BUNDLE_CONTRACT_NATIVE_SOURCES}
        ${GFX_OPENCL_REMOTE_TENSOR_CONTRACT_NATIVE_SOURCES})
else()
    target_sources(${UNIT_TARGET_NAME} PRIVATE
        ${GFX_OPENCL_DEVICE_PROFILE_CONTRACT_UNAVAILABLE_SOURCES}
        ${GFX_OPENCL_RUNTIME_BUNDLE_CONTRACT_UNAVAILABLE_SOURCES}
        ${GFX_OPENCL_REMOTE_TENSOR_CONTRACT_UNAVAILABLE_SOURCES})
endif()
if(APPLE)
    target_sources(${UNIT_TARGET_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/backends/metal/set_device_name.cpp
        ${GFX_METAL_NATIVE_CONTRACT_NATIVE_SOURCES})
else()
    target_sources(${UNIT_TARGET_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/backends/metal/set_device_name_stub.cpp
        ${GFX_METAL_NATIVE_CONTRACT_UNAVAILABLE_SOURCES}
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/backends/metal/runtime/dtype.cpp)
endif()
target_link_libraries(${UNIT_TARGET_NAME} PRIVATE ${GFX_UNIT_TEST_LINK_LIBS})
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
add_test(NAME ${UNIT_TARGET_NAME} COMMAND ${UNIT_TARGET_NAME})
set_tests_properties(${UNIT_TARGET_NAME} PROPERTIES
    LABELS "OV;GFX"
    ENVIRONMENT "${GFX_TEST_RUNTIME_ENVIRONMENT}")
add_dependencies(${UNIT_TARGET_NAME} gfx_test_plugins_xml)

add_executable(${RUNTIME_MICRO_TARGET_NAME})
add_dependencies(${RUNTIME_MICRO_TARGET_NAME} ${GFX_TEST_DEPENDENCIES})
add_dependencies(${RUNTIME_MICRO_TARGET_NAME} gfx_test_plugins_xml)
set(GFX_RUNTIME_EXECUTION_POLICY_SOURCE
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/gfx_runtime_execution_fork.cpp)
if(APPLE OR WIN32 OR ANDROID OR GFX_BACKEND_OPENCL_AVAILABLE)
    set(GFX_RUNTIME_EXECUTION_POLICY_SOURCE
        ${CMAKE_CURRENT_SOURCE_DIR}/unit/gfx_runtime_execution_inline.cpp)
endif()
target_sources(${RUNTIME_MICRO_TARGET_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/gfx_runtime_model_runner.cpp
    ${GFX_RUNTIME_EXECUTION_POLICY_SOURCE}
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/gfx_activation_runtime_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/gfx_add_runtime_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/gfx_matmul_runtime_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/gfx_multiply_runtime_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/gfx_reduce_logical_runtime_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/gfx_split_runtime_test.cpp)
target_link_libraries(${RUNTIME_MICRO_TARGET_NAME} PRIVATE
    openvino::runtime::dev
    openvino::runtime
    common_test_utils
    gmock
    gtest_main
    gtest)
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
set_tests_properties(${RUNTIME_MICRO_TARGET_NAME} PROPERTIES
    LABELS "OV;GFX"
    ENVIRONMENT "${GFX_TEST_RUNTIME_ENVIRONMENT}")

set(GFX_GTEST_MATRIX_OUTPUT_DIR
    "${CMAKE_CURRENT_BINARY_DIR}/gtest-matrix")
set(GFX_GTEST_MATRIX_CAPTURE_ARGS
    "--capture"
        "${TARGET_NAME}::$<TARGET_FILE:${TARGET_NAME}>::${GFX_GTEST_MATRIX_OUTPUT_DIR}/${TARGET_NAME}.txt"
    "--capture"
        "${UNIT_TARGET_NAME}::$<TARGET_FILE:${UNIT_TARGET_NAME}>::${GFX_GTEST_MATRIX_OUTPUT_DIR}/${UNIT_TARGET_NAME}.txt"
    "--capture"
        "${RUNTIME_MICRO_TARGET_NAME}::$<TARGET_FILE:${RUNTIME_MICRO_TARGET_NAME}>::${GFX_GTEST_MATRIX_OUTPUT_DIR}/${RUNTIME_MICRO_TARGET_NAME}.txt")
set(GFX_GTEST_MATRIX_TARGETS
    ${TARGET_NAME}
    ${UNIT_TARGET_NAME}
    ${RUNTIME_MICRO_TARGET_NAME})
set(GFX_GTEST_MATRIX_COMPARE_TARGET_ARGS)
foreach(_gfx_gtest_matrix_target IN LISTS GFX_GTEST_MATRIX_TARGETS)
    list(APPEND GFX_GTEST_MATRIX_COMPARE_TARGET_ARGS
        --target "${_gfx_gtest_matrix_target}")
endforeach()
set(GFX_GTEST_MATRIX_REFERENCE_ARGS)
foreach(_gfx_gtest_matrix_root IN LISTS GFX_GTEST_MATRIX_REFERENCE_ROOTS)
    list(APPEND GFX_GTEST_MATRIX_REFERENCE_ARGS
        --root "${_gfx_gtest_matrix_root}")
endforeach()

set(GFX_GTEST_MATRIX_CAPTURE_ON_HOST TRUE)
if(CMAKE_CROSSCOMPILING AND NOT CMAKE_CROSSCOMPILING_EMULATOR)
    set(GFX_GTEST_MATRIX_CAPTURE_ON_HOST FALSE)
endif()

if(GFX_GTEST_MATRIX_CAPTURE_ON_HOST)
    add_custom_target(gfx_gtest_matrix_capture
        COMMAND "${CMAKE_COMMAND}" -E env ${GFX_TEST_RUNTIME_ENVIRONMENT}
            "${Python3_EXECUTABLE}"
            "${GFX_GTEST_MATRIX_TOOL}"
            --check-only
            ${GFX_GTEST_MATRIX_CAPTURE_ARGS}
        DEPENDS
            ${TARGET_NAME}
            ${UNIT_TARGET_NAME}
            ${RUNTIME_MICRO_TARGET_NAME}
            gfx_test_plugins_xml
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        COMMENT "Capture GFX production --gtest_list_tests registrations"
        VERBATIM)

    add_test(
        NAME ov_gfx_gtest_matrix_capture
        COMMAND "${CMAKE_COMMAND}" -E env ${GFX_TEST_RUNTIME_ENVIRONMENT}
            "${Python3_EXECUTABLE}"
            "${GFX_GTEST_MATRIX_TOOL}"
            --check-only
            ${GFX_GTEST_MATRIX_CAPTURE_ARGS})
    set_tests_properties(ov_gfx_gtest_matrix_capture PROPERTIES
        LABELS "OV;GFX"
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

    if(GFX_GTEST_MATRIX_REFERENCE_ARGS)
        add_custom_target(gfx_gtest_matrix_compare
            COMMAND "${CMAKE_COMMAND}" -E env ${GFX_TEST_RUNTIME_ENVIRONMENT}
                "${Python3_EXECUTABLE}"
                "${GFX_GTEST_MATRIX_TOOL}"
                --root "current=${GFX_GTEST_MATRIX_OUTPUT_DIR}"
                ${GFX_GTEST_MATRIX_REFERENCE_ARGS}
                ${GFX_GTEST_MATRIX_COMPARE_TARGET_ARGS}
            DEPENDS gfx_gtest_matrix_capture
            WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
            COMMENT "Compare GFX production --gtest_list_tests registrations across captured targets"
            VERBATIM)

        add_test(
            NAME ov_gfx_gtest_matrix_compare
            COMMAND "${CMAKE_COMMAND}" -E env ${GFX_TEST_RUNTIME_ENVIRONMENT}
                "${Python3_EXECUTABLE}"
                "${GFX_GTEST_MATRIX_TOOL}"
                --root "current=${GFX_GTEST_MATRIX_OUTPUT_DIR}"
                ${GFX_GTEST_MATRIX_REFERENCE_ARGS}
                ${GFX_GTEST_MATRIX_COMPARE_TARGET_ARGS})
        set_tests_properties(ov_gfx_gtest_matrix_compare PROPERTIES
            LABELS "OV;GFX"
            WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
            DEPENDS ov_gfx_gtest_matrix_capture)
    else()
        add_custom_target(gfx_gtest_matrix_compare
            COMMAND "${CMAKE_COMMAND}" -E echo
                "GFX: cross-target gtest matrix comparison requires GFX_GTEST_MATRIX_REFERENCE_ROOTS entries for macos, android, rpi4 and rpi5"
            COMMAND "${CMAKE_COMMAND}" -E false
            DEPENDS gfx_gtest_matrix_capture
            WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
            COMMENT "Reject incomplete GFX cross-target --gtest_list_tests comparison"
            VERBATIM)
    endif()
else()
    add_custom_target(gfx_gtest_matrix_capture
        COMMAND "${CMAKE_COMMAND}" -E echo
            "GFX: host --gtest_list_tests capture is unavailable for this cross-build; run the test binaries on the target device or configure CMAKE_CROSSCOMPILING_EMULATOR"
        COMMAND "${CMAKE_COMMAND}" -E false
        COMMENT "Reject host-only GFX gtest matrix capture for cross-build"
        VERBATIM)

    add_custom_target(gfx_gtest_matrix_compare
        COMMAND "${CMAKE_COMMAND}" -E echo
            "GFX: cross-target gtest matrix comparison requires device-captured reference roots for macos, android, rpi4 and rpi5"
        COMMAND "${CMAKE_COMMAND}" -E false
        COMMENT "Reject incomplete GFX cross-target --gtest_list_tests comparison"
        VERBATIM)
endif()

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

add_executable(ov_gfx_conv_shape_bench
    ${CMAKE_CURRENT_SOURCE_DIR}/tools/ov_gfx_conv_shape_bench.cpp
)
target_link_libraries(ov_gfx_conv_shape_bench PRIVATE
    openvino::runtime
    common_test_utils
)
if(TARGET gfx_plugin_build_settings)
    target_link_libraries(ov_gfx_conv_shape_bench PRIVATE gfx_plugin_build_settings)
endif()

if(ANDROID)
    add_executable(ov_gfx_opencl_conv_microbench_android
        ${CMAKE_CURRENT_SOURCE_DIR}/tools/ov_gfx_opencl_conv_microbench_android.cpp
    )
    target_link_libraries(ov_gfx_opencl_conv_microbench_android PRIVATE dl)
    if(TARGET gfx_plugin_build_settings)
        target_link_libraries(ov_gfx_opencl_conv_microbench_android PRIVATE gfx_plugin_build_settings)
    endif()
endif()

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
elseif(GFX_BACKEND_OPENCL_AVAILABLE)
    target_link_libraries(ov_gfx_microbench PRIVATE gfx_runtime_opencl)
endif()
if(TARGET GFX::Metal)
    target_link_libraries(ov_gfx_microbench PRIVATE GFX::Metal)
endif()
if(TARGET gfx_plugin_build_settings)
    target_link_libraries(ov_gfx_microbench PRIVATE gfx_plugin_build_settings)
endif()
target_include_directories(ov_gfx_microbench PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../src
    ${CMAKE_CURRENT_BINARY_DIR}/../src
    ${OpenVINO_SOURCE_DIR}/src
    ${OpenVINO_SOURCE_DIR}/src/plugins/template/include
)

# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "armPlugin")

set(CMAKE_CXX_STANDARD 14)

file(GLOB_RECURSE SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB_RECURSE HEADERS
     ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)

ie_arm_neon_optimization_flags(neon_flags)
set_source_files_properties(${SOURCES} PROPERTIES COMPILE_FLAGS "${neon_flags}")

# adds a shared library with plugin
ie_add_plugin(NAME ${TARGET_NAME}
              DEVICE_NAME "CPU"
              SOURCES ${SOURCES} ${HEADERS}
              SKIP_INSTALL # ATTENTION: uncomment to install component
              VERSION_DEFINES_FOR arm_plugin.cpp)

target_include_directories(${TARGET_NAME} PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${IE_MAIN_ARM_PLUGIN_SOURCE_DIR}/include")

target_link_libraries(${TARGET_NAME} PRIVATE
    IE::inference_engine
    arm_compute::arm_compute
    arm_compute::arm_compute_core
    arm_compute::half
    ${NGRAPH_LIBRARIES}
    IE::inference_engine_transformations
    IE::ngraph_reference
)

if(NOT DEFINED ENABLE_LTO)
    set(ENABLE_LTO OFF)
endif()
set_target_properties(${TARGET_NAME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})

ie_register_plugins(MAIN_TARGET ${TARGET_NAME}
                    POSSIBLE_PLUGINS ${TARGET_NAME})

# ATTENTION: uncomment to install component
# install

# set(component_name armPlugin)
# ie_cpack_add_component(${component_name} REQUIRED)

# install(TARGETS ${TARGET_NAME}
#         RUNTIME DESTINATION ${IE_CPACK_LIBRARY_PATH}
#         ARCHIVE DESTINATION ${IE_CPACK_LIBRARY_PATH}
#         LIBRARY DESTINATION ${IE_CPACK_LIBRARY_PATH}
#         COMPONENT ${component_name})

# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "CUDAPlugin")

file(GLOB_RECURSE SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/*.cu)
file(GLOB_RECURSE HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp ${CMAKE_CURRENT_SOURCE_DIR}/*.cuh)

set_source_files_properties(*.cu  PROPERTIES LANGUAGE CUDA)
set_source_files_properties(*.cuh PROPERTIES LANGUAGE CUDA)

# Adds a shared library with plugin
ie_add_plugin(NAME ${TARGET_NAME}
              DEVICE_NAME "CUDA"
              SOURCES ${SOURCES} ${HEADERS}
              SKIP_INSTALL # ATTENTION: uncomment to install component
              VERSION_DEFINES_FOR cuda_plugin.cpp)
# Adds test object library
add_library(${TARGET_NAME}_test_static STATIC EXCLUDE_FROM_ALL ${SOURCES} ${HEADERS})
if (${CUDAToolkit_VERSION_MAJOR} LESS_EQUAL "10")
    set_property(TARGET ${TARGET_NAME} PROPERTY CUDA_ARCHITECTURES 30 35 50 60 72)
    set_property(TARGET ${TARGET_NAME}_test_static PROPERTY CUDA_ARCHITECTURES 30 35 50 60 72)
else()
    set_property(TARGET ${TARGET_NAME} PROPERTY CUDA_ARCHITECTURES 35 50 60 72)
    set_property(TARGET ${TARGET_NAME}_test_static PROPERTY CUDA_ARCHITECTURES 35 50 60 72)
endif()

target_include_directories(${TARGET_NAME}
                           PUBLIC
                           "${CUDAToolkit_INCLUDE_DIRS}"
                           PRIVATE
                           "${CMAKE_CURRENT_SOURCE_DIR}"
                           "${IE_MAIN_CUDA_PLUGIN_SOURCE_DIR}/include"
                           "${InferenceEngine_C_API_SOURCE_DIR}/include")
target_include_directories(${TARGET_NAME}_test_static
                           PUBLIC
                           "${CUDAToolkit_INCLUDE_DIRS}"
                           PRIVATE
                           "${CMAKE_CURRENT_SOURCE_DIR}"
                           "${IE_MAIN_CUDA_PLUGIN_SOURCE_DIR}/include"
                           "${InferenceEngine_C_API_SOURCE_DIR}/include")

# Link common Inference Engine libraries
target_link_libraries(${TARGET_NAME}
                      PRIVATE
                      IE::inference_engine
                      IE::inference_engine_transformations
                      GSL
                      fmt-header-only
                      CUDA::cudart
                      CUDA::cuda_driver
                      ${NGRAPH_LIBRARIES})
target_link_libraries(${TARGET_NAME}_test_static
                      PRIVATE
                      IE::inference_engine
                      IE::inference_engine_transformations
                      GSL
                      fmt-header-only
                      CUDA::cudart
                      CUDA::cuda_driver
                      ${NGRAPH_LIBRARIES})
if(TARGET IE::inference_engine_plugin_api)
    target_link_libraries(${TARGET_NAME}_test_static PRIVATE IE::inference_engine_plugin_api)
else()
    target_link_libraries(${TARGET_NAME}_test_static PRIVATE inference_engine_plugin_api)
endif()

# Link inference backend library to plugin. Here we use ngraph interpreter_backend as example
target_link_libraries(${TARGET_NAME}
                      PRIVATE
                      IE::ngraph_backend
                      IE::interpreter_backend
                      CUDA::cudart)
target_link_libraries(${TARGET_NAME}_test_static
                      PRIVATE
                      IE::ngraph_backend
                      IE::interpreter_backend
                      CUDA::cudart)

set_target_properties(${TARGET_NAME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})


# ATTENTION: uncomment to register a plugin in the plugins.xml file
if(ENABLE_CUDA)
    ie_register_plugins(MAIN_TARGET ${TARGET_NAME}
                        POSSIBLE_PLUGINS ${TARGET_NAME})
endif()

# ATTENTION: uncomment to install component
# install

# set(component_name template)
# ie_cpack_add_component(${component_name} REQUIRED)

# install(TARGETS ${TARGET_NAME}
#         RUNTIME DESTINATION ${IE_CPACK_RUNTIME_PATH}
#         ARCHIVE DESTINATION ${IE_CPACK_ARCHIVE_PATH}
#         LIBRARY DESTINATION ${IE_CPACK_LIBRARY_PATH}
#         COMPONENT ${component_name})

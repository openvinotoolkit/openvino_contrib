# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME CUDAPlugin)
set(OBJ_NAME ${TARGET_NAME}_OBJ)

file(GLOB_RECURSE SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/*.cu)
file(GLOB_RECURSE HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp ${CMAKE_CURRENT_SOURCE_DIR}/*.cuh)

set_source_files_properties(*.cu *.cuh PROPERTIES LANGUAGE CUDA)

add_library(${OBJ_NAME} STATIC ${SOURCES})

# Adds a shared library with plugin
ie_add_plugin(NAME ${TARGET_NAME}
              DEVICE_NAME "CUDA"
              SOURCES ${HEADERS}
              SKIP_INSTALL # ATTENTION: uncomment to install component
              VERSION_DEFINES_FOR cuda_plugin.cpp)
if (${CUDAToolkit_VERSION_MAJOR} LESS_EQUAL "10")
    set_property(TARGET ${OBJ_NAME} PROPERTY CUDA_ARCHITECTURES 30 35 50 60 72)
else()
    set_property(TARGET ${OBJ_NAME} PROPERTY CUDA_ARCHITECTURES 35 50 60 72)
endif()

ieTargetLinkWholeArchive(${TARGET_NAME} ${OBJ_NAME})

target_include_directories(${OBJ_NAME}
                           SYSTEM PUBLIC
                           "${CUDAToolkit_INCLUDE_DIRS}"
)
target_include_directories(${OBJ_NAME}
                           PUBLIC
                           "${CUTENSOR_INCLUDE_DIR}"
                           "${CMAKE_CURRENT_SOURCE_DIR}"
                           "${IE_MAIN_CUDA_PLUGIN_SOURCE_DIR}/include"
                           "${InferenceEngine_C_API_SOURCE_DIR}/include"
)

target_link_libraries(${OBJ_NAME}
                      PUBLIC
                      GSL
                      fmt-header-only
                      IE::ngraph_backend
                      IE::inference_engine
                      IE::inference_engine_plugin_api
                      PRIVATE
                      CUDA::cudart
                      CUDA::cublas
                      CUDA::cudnn
                      CUDA::cutensor
                      IE::inference_engine_transformations
                      IE::interpreter_backend
                      ${NGRAPH_LIBRARIES}
)

set_target_properties(${OBJ_NAME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})


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

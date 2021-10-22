# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME CUDAPlugin)
set(OBJ_NAME ${TARGET_NAME}_OBJ)

set(SOURCE_MASKS *.cpp *.cu)
set(HEADER_MASKS *.hpp *.cuh)
if(${CUDAToolkit_VERSION_MAJOR} EQUAL 10)
    set(SOURCE_MASKS *.cpp split.cu concat.cu strided_slice.cu gather.cu select.cu
        insert.cu slice.cu logical_not.cu elementwise.cu comparison.cu)
    set(HEADER_MASKS *.hpp)
endif(${CUDAToolkit_VERSION_MAJOR} EQUAL 10)

file(GLOB_RECURSE
	SOURCES
	RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
	CONFIGURE_DEPENDS
	${SOURCE_MASKS}
)
list(REMOVE_ITEM SOURCES cuda_create_plugin.cpp)
list(FILTER SOURCES EXCLUDE REGEX "^ops/examples/.*$")
file(GLOB_RECURSE
	HEADERS
	RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
	${HEADER_MASKS}
)

set_source_files_properties(*.cu *.cuh PROPERTIES LANGUAGE CUDA)

add_library(${OBJ_NAME} STATIC ${SOURCES})
target_compile_definitions(${OBJ_NAME} PRIVATE IMPLEMENT_INFERENCE_ENGINE_PLUGIN)

# Adds a shared library with plugin
ie_add_plugin(NAME ${TARGET_NAME}
              DEVICE_NAME "CUDA"
              SOURCES ${HEADERS} cuda_create_plugin.cpp
              SKIP_INSTALL # ATTENTION: uncomment to install component
              VERSION_DEFINES_FOR cuda_create_plugin.cpp)
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
                           $<TARGET_PROPERTY:IE::inference_engine_transformations,INTERFACE_INCLUDE_DIRECTORIES>
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

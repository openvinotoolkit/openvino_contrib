# Copyright (C) 2020 Intel Corporation

project(inference_engine_java_api)

if (NOT CMAKE_CROSSCOMPILING)
    set(JAVA_AWT_LIBRARY NotNeeded)
    set(JAVA_JVM_LIBRARY NotNeeded)
    set(JAVA_AWT_INCLUDE_PATH NotNeeded)
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
    find_package(JNI REQUIRED)
endif()

# Build native part

if(DEFINED OpenVINO_SOURCE_DIR)
    set(InferenceEngine_LIBRARIES inference_engine)
else()
    find_package(InferenceEngineDeveloperPackage QUIET)
    if(NOT InferenceEngineDeveloperPackage_FOUND)
      find_package(InferenceEngine REQUIRED)
    endif()
    set(InferenceEngine_LIBRARIES IE::inference_engine)
endif()

file(GLOB_RECURSE sources ${CMAKE_CURRENT_SOURCE_DIR}/cpp/*.cpp)

add_library(${PROJECT_NAME} SHARED ${sources})

target_link_libraries(${PROJECT_NAME} PRIVATE ${InferenceEngine_LIBRARIES})
target_include_directories(${PROJECT_NAME} PRIVATE ${JNI_INCLUDE_DIRS})

install(TARGETS ${PROJECT_NAME} EXPORT OpenVINOTargets
        RUNTIME DESTINATION ${IE_CPACK_RUNTIME_PATH} COMPONENT java_api
        ARCHIVE DESTINATION ${IE_CPACK_ARCHIVE_PATH} COMPONENT java_api
        LIBRARY DESTINATION ${IE_CPACK_LIBRARY_PATH} COMPONENT java_api)

# if(ENABLE_SAMPLES)
#     add_subdirectory(samples)
# endif()

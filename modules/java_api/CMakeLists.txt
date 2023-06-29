# Copyright (C) 2020-2023 Intel Corporation

project(inference_engine_java_api)

if (NOT CMAKE_CROSSCOMPILING)
    set(JAVA_AWT_LIBRARY NotNeeded)
    set(JAVA_JVM_LIBRARY NotNeeded)
    set(JAVA_AWT_INCLUDE_PATH NotNeeded)
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
    find_package(JNI REQUIRED)
endif()

# Build native part

find_package(OpenVINODeveloperPackage QUIET
             PATHS "${InferenceEngineDeveloperPackage_DIR}")
if(NOT OpenVINODeveloperPackage_FOUND)
    find_package(OpenVINO REQUIRED)
endif()

file(GLOB_RECURSE sources ${CMAKE_CURRENT_SOURCE_DIR}/src/main/cpp/*.cpp)

add_library(${PROJECT_NAME} SHARED ${sources})

target_link_libraries(${PROJECT_NAME} PRIVATE openvino::runtime)
target_include_directories(${PROJECT_NAME} PRIVATE ${JNI_INCLUDE_DIRS})
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/main/cpp)

install(TARGETS ${PROJECT_NAME}
        RUNTIME DESTINATION ${IE_CPACK_RUNTIME_PATH} COMPONENT java_api
        ARCHIVE DESTINATION ${IE_CPACK_ARCHIVE_PATH} COMPONENT java_api
        LIBRARY DESTINATION ${IE_CPACK_LIBRARY_PATH} COMPONENT java_api)

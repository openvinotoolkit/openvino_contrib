# Copyright (C) 2020 Intel Corporation

# Find OpenCV components if exist

find_package(OpenCV QUIET)
find_package(Java REQUIRED)
include(UseJava)
find_package(JNI REQUIRED)

if(EXISTS "${OpenCV_INSTALL_PATH}/share/java")
    file(GLOB_RECURSE JAVA_OPENCV_SRC ${OpenCV_INSTALL_PATH}/share/java/*.jar)
elseif(EXISTS "${OpenCV_INSTALL_PATH}/bin")
    file(GLOB JAVA_OPENCV_SRC ${OpenCV_INSTALL_PATH}/bin/*.jar)
endif()

#
# ie_add_java_sample(NAME <target name>
#               SOURCES <source files>
#               [DEPENDENCIES <dependencies>]
#               [OPENCV_DEPENDENCIES]
#
macro(ie_add_java_sample)
    set(options OPENCV_DEPENDENCIES)
    set(oneValueArgs NAME)
    cmake_parse_arguments(OV_SAMPLE "${options}" "${oneValueArgs}"
                          "" ${ARGN} )

    # Collect sample sources

    file(GLOB_RECURSE SAMPLE_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.java)
    list(APPEND SAMPLE_SOURCES ${PROJECT_SOURCE_DIR}/ArgumentParser.java)

    # Set OpenVINO component

    set(OV_JAVA_SRC $ENV{OV_JAVA_DIR}/java_api.jar)

    # Add OpenCV components if required

    if(OV_SAMPLE_OPENCV_DEPENDENCIES) 
        if(EXISTS "${JAVA_OPENCV_SRC}")
            list(APPEND OV_JAVA_SRC ${JAVA_OPENCV_SRC})
        else()
            message(WARNING "OPENCV .jar file is disabled or not found, ${OV_SAMPLE_NAME} skipped")
            return()
        endif()
    endif()

    # Create .jar file from sources
    add_jar("${OV_SAMPLE_NAME}_jar"
                    SOURCES ${SAMPLE_SOURCES}
                    OUTPUT_NAME ${OV_SAMPLE_NAME}  
                    OUTPUT_DIR ${CMAKE_BINARY_DIR}/jars 
                    INCLUDE_JARS ${OV_JAVA_SRC})
          
    add_dependencies(${OV_SAMPLE_NAME}_jar ${OV_JAVA_SRC})

endmacro()

# Collect all samples subdirectories

file(GLOB samples_dirs RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *)

foreach(dir ${samples_dirs})
    if(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${dir})
            # Include subdirectory to the project.
            add_subdirectory(${dir})
    endif() 
endforeach()

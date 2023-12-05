# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.27.9)

# Reject any attempt to use a toolchain file. We must not use one because
# we could be downloading it here. If the CMAKE_TOOLCHAIN_FILE environment
# variable is set, the cache variable will have been initialized from it.
unset(CMAKE_TOOLCHAIN_FILE CACHE)
unset(ENV{CMAKE_TOOLCHAIN_FILE})

# We name the project and the target for the ExternalProject_Add() call
# to something that will highlight to the user what we are working on if
# something goes wrong and an error message is produced.

project(sentencepiece-populate NONE)



include(ExternalProject)
ExternalProject_Add(sentencepiece-populate
                     "UPDATE_DISCONNECTED" "False" "DOWNLOAD_EXTRACT_TIMESTAMP" "FALSE" "EXTERNALPROJECT_INTERNAL_ARGUMENT_SEPARATOR" "URL" "https://github.com/google/sentencepiece/archive/refs/tags/v0.1.99.tar.gz" "URL_HASH" "SHA256=63617eaf56c7a3857597dcd8780461f57dd21381b56a27716ef7d7e02e14ced4"
                    SOURCE_DIR          "/home/rmikhail/src/openvino_contrib/modules/custom_operations/_skbuild/linux-x86_64-3.11/cmake-build/_deps/sentencepiece-src"
                    BINARY_DIR          "/home/rmikhail/src/openvino_contrib/modules/custom_operations/_skbuild/linux-x86_64-3.11/cmake-build/_deps/sentencepiece-build"
                    CONFIGURE_COMMAND   ""
                    BUILD_COMMAND       ""
                    INSTALL_COMMAND     ""
                    TEST_COMMAND        ""
                    USES_TERMINAL_DOWNLOAD  YES
                    USES_TERMINAL_UPDATE    YES
                    USES_TERMINAL_PATCH     YES
)



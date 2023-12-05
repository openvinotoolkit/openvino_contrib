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

project(fast_tokenizer-populate NONE)



include(ExternalProject)
ExternalProject_Add(fast_tokenizer-populate
                     "UPDATE_DISCONNECTED" "False" "DOWNLOAD_EXTRACT_TIMESTAMP" "FALSE" "EXTERNALPROJECT_INTERNAL_ARGUMENT_SEPARATOR" "URL" "https://bj.bcebos.com/paddlenlp/fast_tokenizer/fast_tokenizer-linux-x64-1.0.2.tgz" "URL_HASH" "SHA256=843a8299b55ef2e06ea50ba0d4ab4cb05b9e4cdb7cb8e29f3d55c494a1b7aecc"
                    SOURCE_DIR          "/home/rmikhail/src/openvino_contrib/modules/custom_operations/_skbuild/linux-x86_64-3.11/cmake-build/_deps/fast_tokenizer-src"
                    BINARY_DIR          "/home/rmikhail/src/openvino_contrib/modules/custom_operations/_skbuild/linux-x86_64-3.11/cmake-build/_deps/fast_tokenizer-build"
                    CONFIGURE_COMMAND   ""
                    BUILD_COMMAND       ""
                    INSTALL_COMMAND     ""
                    TEST_COMMAND        ""
                    USES_TERMINAL_DOWNLOAD  YES
                    USES_TERMINAL_UPDATE    YES
                    USES_TERMINAL_PATCH     YES
)



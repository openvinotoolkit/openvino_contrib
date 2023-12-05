# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/home/rmikhail/src/openvino_contrib/modules/custom_operations/_skbuild/linux-x86_64-3.11/cmake-build/_deps/fast_tokenizer-src"
  "/home/rmikhail/src/openvino_contrib/modules/custom_operations/_skbuild/linux-x86_64-3.11/cmake-build/_deps/fast_tokenizer-build"
  "/home/rmikhail/src/openvino_contrib/modules/custom_operations/_skbuild/linux-x86_64-3.11/cmake-build/_deps/fast_tokenizer-subbuild/fast_tokenizer-populate-prefix"
  "/home/rmikhail/src/openvino_contrib/modules/custom_operations/_skbuild/linux-x86_64-3.11/cmake-build/_deps/fast_tokenizer-subbuild/fast_tokenizer-populate-prefix/tmp"
  "/home/rmikhail/src/openvino_contrib/modules/custom_operations/_skbuild/linux-x86_64-3.11/cmake-build/_deps/fast_tokenizer-subbuild/fast_tokenizer-populate-prefix/src/fast_tokenizer-populate-stamp"
  "/home/rmikhail/src/openvino_contrib/modules/custom_operations/_skbuild/linux-x86_64-3.11/cmake-build/_deps/fast_tokenizer-subbuild/fast_tokenizer-populate-prefix/src"
  "/home/rmikhail/src/openvino_contrib/modules/custom_operations/_skbuild/linux-x86_64-3.11/cmake-build/_deps/fast_tokenizer-subbuild/fast_tokenizer-populate-prefix/src/fast_tokenizer-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/rmikhail/src/openvino_contrib/modules/custom_operations/_skbuild/linux-x86_64-3.11/cmake-build/_deps/fast_tokenizer-subbuild/fast_tokenizer-populate-prefix/src/fast_tokenizer-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/rmikhail/src/openvino_contrib/modules/custom_operations/_skbuild/linux-x86_64-3.11/cmake-build/_deps/fast_tokenizer-subbuild/fast_tokenizer-populate-prefix/src/fast_tokenizer-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()

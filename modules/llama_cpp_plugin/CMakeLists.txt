cmake_minimum_required(VERSION 3.13)

project(LlamaCppPlugin)

find_package(OpenVINODeveloperPackage REQUIRED)

ov_option(ENABLE_LLAMA_CPP_PLUGIN_REGISTRATION "Enables registration of LLAMA_CPP plugin" OFF)

if(CMAKE_COMPILER_IS_GNUCXX)
    ov_add_compiler_flags(-Wall)
endif()

add_subdirectory(src)
add_subdirectory(tools)

add_subdirectory(third_party/llama.cpp)

if(ENABLE_TESTS)
    include(CTest)
    enable_testing()

    if(ENABLE_FUNCTIONAL_TESTS)
        add_subdirectory(tests/functional)
    endif()
endif()


# install

if(OpenVINODeveloperPackage_FOUND)
    ov_cpack(LlamaCppPlugin)
endif()

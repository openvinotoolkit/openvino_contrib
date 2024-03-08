set( PLUGIN_LIBRARY_NAME CACHE STRING "Library name for the generated plugin" ${TARGET_NAME})
if(NOT PLUGIN_LIBRARY_NAME)
    set( PLUGIN_LIBRARY_NAME "llama_cpp_plugin" )
endif()

set( PLUGIN_DEVICE_NAME CACHE STRING "Device name for the resulting plugin")
if(NOT PLUGIN_DEVICE_NAME)
    set( PLUGIN_DEVICE_NAME "LLAMA_CPP" )
endif()

set(TARGET_NAME ${PLUGIN_LIBRARY_NAME})

file(GLOB_RECURSE SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB_RECURSE HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)

if (NOT ENABLE_TEMPLATE_REGISTRATION)
    # Skip install and registration of template component
    set(skip_plugin SKIP_INSTALL SKIP_REGISTRATION)
endif()



# adds a shared library with plugin
ov_add_plugin(NAME ${TARGET_NAME}
              DEVICE_NAME ${PLUGIN_DEVICE_NAME}
              SOURCES ${SOURCES} ${HEADERS}
              ${skip_plugin}
              VERSION_DEFINES_FOR plugin.cpp
              ADD_CLANG_FORMAT)

# Enable support of CC for the plugin
ov_mark_target_as_cc(${TARGET_NAME})

target_include_directories(${TARGET_NAME} PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${LlamaCppPlugin_SOURCE_DIR}/include")

# link common OpenVINO Runtime libraries
target_link_libraries(${TARGET_NAME} PRIVATE
    openvino::interpreter_backend
    openvino::reference)

set( LLAMA_TARGET_NAME CACHE STRING "Exact target exposed by llama.cpp to link against as the main llama.cpp library")
if(NOT LLAMA_TARGET_NAME)
    set( LLAMA_TARGET_NAME "llama" )
endif()

# include and link llama.cpp  and ggml code
target_link_libraries(${TARGET_NAME} PRIVATE ${LLAMA_TARGET_NAME})
target_link_libraries(${TARGET_NAME} PRIVATE ggml)


set_target_properties(${TARGET_NAME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})

if (ENABLE_TEMPLATE_REGISTRATION)
    # Update the plugins.xml file
    ov_register_plugins(MAIN_TARGET ${TARGET_NAME})
endif()

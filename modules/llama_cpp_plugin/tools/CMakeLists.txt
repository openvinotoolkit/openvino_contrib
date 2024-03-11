cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 11)

find_package(OpenVINO REQUIRED)


add_executable(llama_cpp_runner
               "${CMAKE_CURRENT_SOURCE_DIR}/runner.cpp"
              )
target_link_libraries(llama_cpp_runner PRIVATE openvino::runtime)


add_executable(tensor_comparator
               "${CMAKE_CURRENT_SOURCE_DIR}/tensor_comparator.cpp"
              )
target_link_libraries(tensor_comparator PRIVATE ggml)

add_executable(cache_embedder
               "${CMAKE_CURRENT_SOURCE_DIR}/cache_embedder.cpp"
              )
set_target_properties(cache_embedder PROPERTIES CXX_STANDARD 17)

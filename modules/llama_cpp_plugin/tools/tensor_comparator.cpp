#include "ggml.h"
#include <cassert>
#include <string>
#include <iostream>
#include <fstream>
#include <numeric>
#include <vector>



int main(int argc, char* argv[]) {
    assert(argc == 3 || argc == 4);
    std::string left_name(argv[1]);
    std::string right_name(argv[2]);

    gguf_init_params left_params; left_params.no_alloc = false; left_params.ctx = nullptr;
    gguf_init_params right_params; left_params.no_alloc = false; right_params.ctx = nullptr;
    gguf_context* left_ctx = gguf_init_from_file(left_name.c_str(), left_params);
    gguf_context* right_ctx = gguf_init_from_file(right_name.c_str(), right_params);

    std::vector<std::string> tensor_names;
    if (argc == 4) tensor_names.push_back(std::string(argv[3]));
    else {
        for (size_t idx = 0; idx < left_ctx->header.n_tensors; idx++) {
            gguf_tensor_info left_tensor_info = left_ctx->infos[idx];
            tensor_names.push_back(left_tensor_info.name.data);
        }
    }

    for (const auto& tensor_name : tensor_names) {


        int left_tensor_idx = gguf_find_tensor(left_ctx, tensor_name.c_str());
        int right_tensor_idx = gguf_find_tensor(right_ctx, tensor_name.c_str());

        size_t left_tensor_offset = gguf_get_tensor_offset(left_ctx, left_tensor_idx) + left_ctx->offset;
        size_t right_tensor_offset = gguf_get_tensor_offset(right_ctx, right_tensor_idx) + right_ctx->offset;

        gguf_tensor_info left_tensor_info = left_ctx->infos[left_tensor_idx];
        gguf_tensor_info right_tensor_info = right_ctx->infos[right_tensor_idx];

        std::cout << "tensor name " << tensor_name << ", byte offsets: " << left_tensor_offset << " (left), " << right_tensor_offset << " (right)" << std::endl;
        std::cout << "tensor name " << tensor_name << ", shape: ";
        for (size_t i = 0; i < left_tensor_info.n_dims; i++) {
            std::cout << left_tensor_info.ne[i] << ",";
        }
        std::cout << " (left), ";

        for (size_t i = 0; i < right_tensor_info.n_dims; i++) {
            std::cout << right_tensor_info.ne[i] << ",";
        }
        std::cout  << " (right) " << std::endl;

        size_t left_tensor_size = std::accumulate(std::begin(left_tensor_info.ne), std::begin(left_tensor_info.ne) + GGML_MAX_DIMS, (size_t) sizeof(float), std::multiplies<size_t>());
        size_t right_tensor_size = std::accumulate(std::begin(right_tensor_info.ne), std::begin(right_tensor_info.ne) + GGML_MAX_DIMS, (size_t) sizeof(float), std::multiplies<size_t>());

        std::cout << "tensor name " << tensor_name << ", size (calculated): " << left_tensor_size << " (left), " << right_tensor_size << " (right)" << std::endl;

        if (left_tensor_size != right_tensor_size) {
            std::cout << "size mismatch (" << left_tensor_size << " left, " << right_tensor_size << "right), exiting" << std::endl;
            exit(-1);
        }

        size_t bytes_compared = 0;

        std::ifstream left_file(left_name, std::ios::binary);
        std::ifstream right_file(right_name, std::ios::binary);

        left_file.seekg(left_tensor_offset);
        right_file.seekg(right_tensor_offset);

        std::cout << "first 10 float values:" << std::endl;
        for (size_t i = 0; i < 10; i++) {
            float left_value; left_file.read((char*) &left_value, sizeof(float));
            float right_value; right_file.read((char*) &right_value, sizeof(float));

            std::cout << left_value <<  " left, " << right_value << " right" << std::endl;
        }

        left_file.seekg(left_tensor_offset);
        right_file.seekg(right_tensor_offset);
        for (size_t i = 0; i < left_tensor_size; i++) {
            char left_byte; left_file.read((char*) &left_byte, sizeof(char));
            char right_byte; right_file.read((char*) &right_byte, sizeof(char));

            if (left_byte != right_byte) {
                std::cout << "byte " << bytes_compared << " mismatch (" << std::hex << +((uint8_t) left_byte) << " left, " << +((uint8_t) right_byte) << " right)" << std::endl;
                std::cout << "offset left " << std::hex << left_tensor_offset + bytes_compared << ", right " << right_tensor_offset + bytes_compared << std::endl;
                exit(-1);
            }
            bytes_compared++;
        }
        std::cout << "tensor contents are identical, bytes compared: " << bytes_compared << std::endl;
    }
}

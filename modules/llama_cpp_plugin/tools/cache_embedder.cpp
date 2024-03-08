#include <fstream>
#include <iostream>
#include <cassert>
#include <string>
#include <stdint.h>
#include <filesystem>

int main(int argc, char* argv[]) {
    assert(argc == 3);
    std::string cache_blob_name = argv[1];
    std::string gguf_file_name = argv[2];

    std::uintmax_t original_file_size = std::filesystem::file_size(cache_blob_name);
    std::fstream cache_io_stream(cache_blob_name, std::ios::binary | std::ios::in | std::ios::out);

    {
        std::string tmp;
        std::getline(cache_io_stream, tmp); // skip the blob header
        std::cout << "skipped header line" << std::endl;
    }

    std::uint64_t data_size = 0;
    cache_io_stream.read(reinterpret_cast<char*>(&data_size), sizeof(data_size));
    std::cout << "skipping IR XML content, size " << data_size << std::endl;
    cache_io_stream.seekp(data_size, std::ios::cur); // skip IR xml content

    cache_io_stream.read(reinterpret_cast<char*>(&data_size), sizeof(data_size));
    std::cout << "skipping IR weight content, size " << data_size << std::endl;
    cache_io_stream.seekp(data_size, std::ios::cur);  // skip IR weight content

    std::streampos pos = cache_io_stream.tellp();
    char magic[4];
    for (size_t i = 0; i < 4; i++) {
        cache_io_stream >> magic[i];
    }

    std::string curr_magic(magic);
    std::cout << "magic at current position is " << curr_magic << std::endl;
    assert(curr_magic == "GGUF");
    cache_io_stream.seekp(pos);

    std::ifstream gguf_input_stream(gguf_file_name, std::ios::binary);
    cache_io_stream << gguf_input_stream.rdbuf();
    std::cout << "gguf content write successful" << std::endl;
    std::uintmax_t final_size = cache_io_stream.tellp();
    cache_io_stream.close();
    if (final_size < original_file_size) {
        std::cout << "cache entry is now smaller (" << final_size << " vs original " << original_file_size << "), truncating" << std::endl;
        std::filesystem::resize_file(cache_blob_name, final_size);
    }

    return 0;
}

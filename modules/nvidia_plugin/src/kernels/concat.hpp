// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include "details/cuda_type_traits.hpp"
#include "details/error.hpp"

namespace ov {
namespace nvidia_gpu {
namespace kernel {

class Concat {
public:
    struct Chunk {
        size_t input;
        size_t offset;
    };

    Concat(Type_t element_type,
           size_t numInputs,
           std::vector<Chunk>&& chunks,
           size_t chunkSize,
           size_t allChunkSize,
           size_t numBlocks,
           size_t threadsPerBlock);
    Concat(Concat&&) = default;
    Concat& operator=(Concat&&) = default;

    void operator()(cudaStream_t stream, const void* chunks, const void* const* src, void* dst) const;

    [[nodiscard]] size_t immutableWbSize() const { return sizeof(Chunk) * chunks_.size(); }
    [[nodiscard]] size_t mutableWbSize() const { return sizeof(float*) * num_inputs_; }
    [[nodiscard]] const void* immutableWbData() const { return chunks_.data(); }

private:
    template <typename T>
    void Call(cudaStream_t stream, const void* chunks, const void* const* src, void* dst) const;

    Type_t element_type_{};
    size_t num_inputs_{};
    std::vector<Chunk> chunks_;
    size_t chunk_size_{};
    size_t all_chunk_size_{};
    size_t num_blocks_{};
    size_t threads_per_block_{};
};

}  // namespace kernel
}  // namespace nvidia_gpu
}  // namespace ov

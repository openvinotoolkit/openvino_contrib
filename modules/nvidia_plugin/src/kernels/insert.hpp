// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include "details/cuda_type_traits.hpp"
#include "details/error.hpp"
#include "details/tensor_helpers.hpp"

namespace ov {
namespace nvidia_gpu {
namespace kernel {

class Insert {
public:
    struct Props {
        Shape<size_t, 5> old_shape{};
        Shape<size_t, 5> new_shape{};
        size_t axe;
    };

    Insert(Type_t element_type, const Props& props, size_t max_threads_per_block);
    Insert(Insert&&) = default;
    Insert& operator=(Insert&&) = default;

    void operator()(cudaStream_t stream, const void* src, void* dst, size_t start) const;

    size_t getImmutableWorkbufferSize() const;
    void setImmutableWorkbuffer(void* immutableBuffer);

    void* getKernel() const;
    size_t getSize() const { return size_; }
    size_t getNumBlocks() const { return num_blocks_; }
    size_t getThreadsPerBlock() const { return threads_per_block_; }
    const Props* getPropsPtr() const { return static_cast<const Props*>(props_ptr_); }

private:
    template <typename T>
    void call(const cudaStream_t stream, const void* src, void* dst, const size_t start) const;

    Type_t element_type_{};
    Props props_{};
    size_t size_{};
    size_t num_blocks_{};
    size_t threads_per_block_{};
    void* props_ptr_{};
};

inline size_t Insert::getImmutableWorkbufferSize() const { return sizeof(props_); }

inline void Insert::setImmutableWorkbuffer(void* immutableBuffer) {
    kernel::throwIfError(
        cudaMemcpyAsync(immutableBuffer, static_cast<const void*>(&props_), sizeof(props_), cudaMemcpyHostToDevice));
    props_ptr_ = immutableBuffer;
}

}  // namespace kernel
}  // namespace nvidia_gpu
}  // namespace ov

// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cuda_type_traits.hpp"
#include "error.hpp"
#include "tensor_helpers.hpp"

namespace CUDAPlugin {
namespace kernel {

class Slice {
public:
    struct Props {
        Shape<size_t, 5> old_shape{};
        Shape<size_t, 5> new_shape{};
        size_t axe;
    };

    Slice(Type_t element_type, const Props& props, size_t max_threads_per_block);
    Slice(Slice&&) = default;
    Slice& operator=(Slice&&) = default;

    void operator()(cudaStream_t stream, const void* src, void* dst, size_t start) const;

    size_t getImmutableWorkbufferSize() const;
    void setImmutableWorkbuffer(void* immutableBuffer);

private:
    template <typename T>
    void call(cudaStream_t stream, const void* src, void* dst, size_t start) const;

    Type_t element_type_{};
    Props props_{};
    size_t size_{};
    unsigned num_blocks_{};
    unsigned threads_per_block_{};
    void* props_ptr_{};
};

inline size_t Slice::getImmutableWorkbufferSize() const { return sizeof(props_); }

inline void Slice::setImmutableWorkbuffer(void* immutableBuffer) {
    kernel::throwIfError(
        cudaMemcpyAsync(immutableBuffer, static_cast<const void*>(&props_), sizeof(props_), cudaMemcpyHostToDevice));
    props_ptr_ = immutableBuffer;
}

}  // namespace kernel
}  // namespace CUDAPlugin

// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cuda_type_traits.hpp"

namespace ov {
namespace nvidia_gpu {
namespace kernel {

class Convert {
public:
    Convert(
        Type_t output_element_type, Type_t input_element_type, size_t size, size_t numBlocks, size_t threadsPerBlock);
    Convert(Convert&&) = default;
    Convert& operator=(Convert&&) = default;

    void operator()(cudaStream_t, void*, const void*) const;
    using convert_t = void (*)(cudaStream_t, size_t, void*, const void*, unsigned, unsigned);

private:
    convert_t convert_kernel_;
    size_t size_;
    size_t num_blocks_;
    size_t threads_per_block_;
};

}  // namespace kernel
}  // namespace nvidia_gpu
}  // namespace ov

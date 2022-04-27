// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include "elementtypeswitch.hpp"
#include "cuda_type_traits.hpp"
#include "numpy_broadcast_mapper.cuh"
#include "error.hpp"

namespace CUDAPlugin {
namespace kernel {

class Broadcast {
public:
    Broadcast(CUDAPlugin::kernel::Type_t element_type, size_t dst_num_elements, size_t max_threads_per_block);

    void operator()(const cudaStream_t stream,
                    const void* src,
                    const NumpyBroadcastMapper& src_mapper,
                    void* dst) const;

private:
    friend AllElementTypesSwitch;

    template <typename T>
    constexpr void case_(cudaStream_t, const void*, const NumpyBroadcastMapper&, void*) const noexcept;

    template <typename T>
    void default_(T t, cudaStream_t, const void*, const NumpyBroadcastMapper&, void*) const noexcept;

private:
    Type_t element_type_;
    size_t dst_num_elements_;
    size_t num_blocks_;
    size_t threads_per_block_;
};

}  // namespace kernel
}  // namespace CUDAPlugin

// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "error.hpp"

namespace ov {
namespace nvidia_gpu {
namespace kernel {

class NumpyBroadcastMapper {
public:
    __host__ NumpyBroadcastMapper()
        : src_strides_{nullptr}, dst_strides_{nullptr}, broadcasted_dims_{nullptr}, shape_rank_{0} {}

    /**
     * @param broadcasted_dims for each dimension, indicates whether a dimension is broadcasted.
     *                         0 - broadcasted, 1 - not broadcasted. Passing other values causes
     *                         undefined behavior.
     * @param src_strides Source tensor strides. Source tensor shape is extended with 1 and has
     *                    the same shape rank as output tensor.
     */
    __host__ NumpyBroadcastMapper(const size_t* src_strides,
                                  const size_t* dst_strides,
                                  const size_t* broadcasted_dims,
                                  size_t shape_rank)
        : src_strides_{src_strides},
          dst_strides_{dst_strides},
          broadcasted_dims_{broadcasted_dims},
          shape_rank_{shape_rank} {
        assertThrow(src_strides_ != 0, "src_strides_ == 0");
        assertThrow(dst_strides_ != 0, "dst_strides_ == 0");
        assertThrow(broadcasted_dims_ != 0, "broadcasted_dims_ == 0");
    }

    __host__ __device__ bool identity() const { return broadcasted_dims_ == nullptr; }

    __device__ unsigned srcIndex(unsigned dst_index) const {
        if (identity()) {
            return dst_index;
        } else {
            unsigned src_idx = 0;
            unsigned i = dst_index;
            for (unsigned r = 0; r < shape_rank_; r++) {
                const unsigned dst_stride = dst_strides_[r];
                const unsigned dst_coord = i / dst_stride;
                i = i % dst_stride;
                const unsigned src_coord = broadcasted_dims_[r] * dst_coord;
                src_idx += src_coord * src_strides_[r];
            }
            return src_idx;
        }
    }

private:
    const size_t* src_strides_;
    const size_t* dst_strides_;
    const size_t* broadcasted_dims_;
    size_t shape_rank_;
};

}  // namespace kernel
}  // namespace nvidia_gpu
}  // namespace ov

// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <cuda_runtime.h>
#include <fmt/format.h>

#include <cassert>

#include "logical_not.cuh"

namespace ov {
namespace nvidia_gpu {
namespace kernel {

template <int Rank>
static inline __global__ void logical_not(const bool* input, bool* output, std::size_t size) {
    // Flat input implies shape of rank 1 with shape[0] = blockDim.x
    std::size_t shape = blockDim.x;
    static_assert(Rank == 1, "Supported Rank is 1 !! For other ranks modify the code");
    auto idx = index<Rank>(&shape, LogicalNot::kElementsPerThread);

    static_assert(LogicalNot::kElementsPerThread == sizeof(unsigned int) * 2,
                  "Elements per thread must be equal uint2 size");

    if (idx + LogicalNot::kElementsPerThread <= size) {
        uint2 inputPacked = *reinterpret_cast<const uint2*>(input + idx);
        uint2 outputPacked;
        outputPacked.x = __vseteq4(inputPacked.x, 0);
        outputPacked.y = __vseteq4(inputPacked.y, 0);
        *reinterpret_cast<uint2*>(output + idx) = outputPacked;
    } else {
#pragma unroll LogicalNot::kElementsPerThread
        for (int i = 0; i < LogicalNot::kElementsPerThread; i++) {
            auto actual_index = idx + i;
            if (actual_index >= size) break;
            output[actual_index] = !input[actual_index];
        }
    }
}

LogicalNot::LogicalNot(const eltwise::KernelExecAttrs& kernelExecAttrs, std::size_t payloadRank, std::size_t len)
    : kernel_exec_attrs_{kernelExecAttrs}, payload_rank_{payloadRank}, len_{len} {}

void LogicalNot::operator()(cudaStream_t stream, const bool* src, bool* dst) const {
    switch (payload_rank_) {
        case 1:
            logical_not<1><<<kernel_exec_attrs_.grid, kernel_exec_attrs_.block, 0, stream>>>(src, dst, len_);
            break;
        default:
            throwIEException(fmt::format("Payload rank {} is not supported.", payload_rank_));
    }
}

}  // namespace kernel
}  // namespace nvidia_gpu
}  // namespace ov

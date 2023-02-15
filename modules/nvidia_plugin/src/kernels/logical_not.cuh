// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

#include "details/eltwise.cuh"
#include "details/error.hpp"

namespace ov {
namespace nvidia_gpu {
namespace kernel {

class LogicalNot {
public:
    LogicalNot(const eltwise::KernelExecAttrs& kernelExecAttrs, std::size_t payloadRank, std::size_t len);

    void operator()(cudaStream_t stream, const bool* src, bool* dst) const;

    static constexpr unsigned kElementsPerThread = 8;
    static constexpr unsigned kWarpsPerBlock = 8;

private:
    eltwise::KernelExecAttrs kernel_exec_attrs_;
    std::size_t payload_rank_;
    std::size_t len_;
};

}  // namespace kernel
}  // namespace nvidia_gpu
}  // namespace ov

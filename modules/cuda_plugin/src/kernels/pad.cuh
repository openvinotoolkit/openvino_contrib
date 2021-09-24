// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstddef>

#include "eltwise.cuh"
#include "ngraph/type/element_type.hpp"

namespace CUDAPlugin {
namespace kernel {

class ConstModePad {
public:
    explicit ConstModePad(eltwise::KernelExecAttrs&& kernelExecAttrs, ngraph::element::Type_t dtype, std::size_t outputRank);
    void operator()(cudaStream_t stream,
                    const void* src,
                    void* dst,
                    const void* begin,
                    const std::size_t* srcShape,
                    const std::size_t* dstShape,
                    const void* padValue) const;

    template <typename T>
    void callKernel(cudaStream_t stream,
                    const void* src,
                    void* dst,
                    const void* begin,
                    const std::size_t* srcShape,
                    const std::size_t* dstShape,
                    const void* padValue) const;

    template <typename T, int PayloadRank>
    void callKernel(cudaStream_t stream,
                    const void* src,
                    void* dst,
                    const void* begin,
                    const std::size_t* srcShape,
                    const std::size_t* dstShape,
                    const void* padValue) const;

    constexpr static unsigned kWarpsPerBlock = 8;
    constexpr static unsigned kElementsPerThread = 1;

private:
    eltwise::KernelExecAttrs kernel_exec_attrs_;
    ngraph::element::Type_t dtype_;
    std::size_t output_rank_;
};

}  // namespace kernel
}  // namespace CUDAPlugin

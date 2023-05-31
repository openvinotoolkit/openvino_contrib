// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda/graph.hpp>

namespace ov {
namespace nvidia_gpu {

struct CudaGraphContext {
    CudaGraphContext(bool useCudaGraph = false) : useCudaGraph_{useCudaGraph} {}
    std::optional<CUDA::GraphExec> graphExec_{};
    bool useCudaGraph_;
};

} // namespace nvidia_gpu
} // namespace ov

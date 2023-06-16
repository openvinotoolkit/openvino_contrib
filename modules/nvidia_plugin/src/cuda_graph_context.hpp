// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda/graph.hpp>
#include <map>
#include <string>

namespace ov {
namespace nvidia_gpu {

struct CudaGraphContext {
    std::optional<CUDA::GraphExec> graphExec{};
    std::optional<CUDA::Graph> graph{};
    std::map<std::string, CUDA::UploadNode> parameterNodes;
    std::map<std::string, CUDA::DownloadNode> resultNodes;
};

} // namespace nvidia_gpu
} // namespace ov

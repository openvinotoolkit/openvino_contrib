// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda/runtime.hpp>

#include "cuda_config.hpp"
#include "openvino/core/model.hpp"

namespace ov {
namespace nvidia_gpu {

class GraphTransformer {
public:
    /**
     * @brief Transform takes an ov::Model and applies all the necessary
     *        CUDA-specific transformations to achieve the maximum optimization of the
     *        model for execution on a CUDA device. The transformations may
     *        include CUDA-specific op fusions and some common OpenVino
     *        transformations as well.
     * @param function a valid shared ptr to a model, represented as an
     *        ov::Model instance.
     * @param config a string-string map of configuration for loading an
     * executable network (e.g. a model); this config influences on what exact
     *        transformations are being applied to the original graph.
     */
    void transform(const CUDA::Device& device, std::shared_ptr<ov::Model>& model, const Configuration& config) const;
};

}  // namespace nvidia_gpu
}  // namespace ov

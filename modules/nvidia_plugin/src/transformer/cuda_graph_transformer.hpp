// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda/runtime.hpp>
#include "openvino/core/model.hpp"

#include "cpp/ie_cnn_network.h"
#include "cuda_config.hpp"

namespace ov {
namespace nvidia_gpu {

class GraphTransformer {
public:
    /**
     * @brief Transform takes an ov::Model and applies common OpenVino
     *        graph transformations.
     * @param function a valid shared ptr to a model, represented as an
     *        ov::Model instance.
     * @param config a string-string map of configuration for loading an
     *  executable network (e.g. a model); this config influences on what exact
     *        transformations are being applied to the original graph.
     */
    void common_transform(const CUDA::Device& device,
                          const std::shared_ptr<ov::Model>& model,
                          const InferenceEngine::InputsDataMap& inputInfoMap,
                          const InferenceEngine::OutputsDataMap& outputsInfoMap,
                          const Configuration& config) const;

    std::shared_ptr<ov::Model> clone_and_export_transform(const CUDA::Device& device,
                                                          const std::shared_ptr<const ov::Model>& model,
                                                          const InferenceEngine::InputsDataMap& inputInfoMap,
                                                          const InferenceEngine::OutputsDataMap& outputsInfoMap,
                                                          const Configuration& config) const;
    /**
     * @brief Transform takes an ov::Model and applies only
     *        CUDA-specific transformations to achieve the maximum optimization of the
     *        model for execution on a CUDA device. The transformations may
     *        includes CUDA-specific op fusions.
     * @param function a valid shared ptr to a model, represented as an
     *        ov::Model instance.
     * @param config a string-string map of configuration for loading an
     * executable network (e.g. a model); this config influences on what exact
     *        transformations are being applied to the original graph.
     */
    void cuda_transform(const CUDA::Device& device,
                        const std::shared_ptr<ov::Model>& model,
                        const Configuration& config) const;

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
    void transform(const CUDA::Device& device,
                   const std::shared_ptr<ov::Model>& model,
                   const InferenceEngine::InputsDataMap& inputInfoMap,
                   const InferenceEngine::OutputsDataMap& outputsInfoMap,
                   const Configuration& config) const;
};

}  // namespace nvidia_gpu
}  // namespace ov

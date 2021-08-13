// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda_operation_base.hpp>

namespace CUDAPlugin {

class GatherOp : public OperationBase {
public:
  GatherOp(const CUDA::CreationContext& context,
           const ngraph::Node& node,
           IndexCollection&& inputIds,
           IndexCollection&& outputIds);
  void Execute(const InferenceRequestContext& context,
               Inputs inputTensors,
               Outputs outputTensors,
               const Workbuffers& workbuffers) override;

private:
    template <typename IndexType>
    void Execute(const InferenceRequestContext& context, Inputs inputs, Outputs outputs);

    size_t element_size_;
    ngraph::element::Type_t indices_type_;
    unsigned num_dicts_;
    unsigned index_range_;
    unsigned data_length_;
    unsigned dict_size_;
    unsigned indices_size_;
    unsigned out_size_;
};

} // namespace CUDAPlugin

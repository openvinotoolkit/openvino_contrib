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
                 const Workbuffers& workbuffers) const override;

private:
    template <typename IndexType>
    void ExecuteByDataType(const InferenceRequestContext& context, Inputs inputs, Outputs outputs) const;

    template <typename DataType, typename IndexType>
    void ExecuteImpl(const InferenceRequestContext& context, Inputs inputs, Outputs outputs) const;

    ngraph::element::Type_t element_type_;
    ngraph::element::Type_t indices_type_;
    unsigned num_dicts_;
    unsigned index_range_;
    unsigned data_length_;
    unsigned indices_size_;
    bool gather_chunks_;
    unsigned blocks_per_grid_;
    unsigned threads_per_block_;
};

} // namespace CUDAPlugin

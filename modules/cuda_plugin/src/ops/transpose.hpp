// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda_operation_base.hpp>
#include <optional>
#include <unordered_map>
#include <vector>
#include <ngraph/node.hpp>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace CUDAPlugin {

class TransposeOp : public OperationCuTensor {
public:
    TransposeOp(const CUDA::CreationContext& context,
             const std::shared_ptr<ngraph::Node>& node,
             std::vector<unsigned>&& inputIds,
             std::vector<unsigned>&& outputIds);
    void Execute(const InferenceRequestContext& context,
                 Inputs inputTensors,
                 Outputs outputTensors,
                 const Workbuffers& workbuffers) override;

private:
    using ExtentsMap = std::unordered_map<int, std::int64_t>;

    static std::vector<std::int64_t> extractInputExtents(const ngraph::Node& node);

    static std::vector<std::int64_t> extractOutputExtents(const ngraph::Node& node);

    static std::vector<int> extractInputMode(std::size_t numDims);

    static std::vector<std::int64_t> extractInputStrides(const ngraph::Node& node);

    static std::vector<std::int64_t> extractOutputStrides(const ngraph::Node& node);

    static ExtentsMap extractExtents(const std::vector<std::int64_t>& input_extents);

    static bool isPermutationTensorSpecified(const ngraph::Node& node);

    static std::optional<std::vector<int>> tryToExtractPermutation(const ngraph::Node& node);

    std::vector<int> permutation(const InferenceRequestContext& context, Inputs inputTensors) const;

    ngraph::element::Type_t extractPermutationElementsType(const ngraph::Node& node);

    template<typename T>
    static std::vector<int> downloadPermutationVector(const InferenceRequestContext& context,
            InferenceEngine::gpu::DevicePointer<const void*>, unsigned numDims);

private:
    std::vector<std::int64_t> inputExtents_;
    std::size_t dimsNumber_;
    std::vector<std::int64_t> outputExtents_;
    std::vector<std::int64_t> inputStrides_;
    std::vector<std::int64_t> outputStrides_;
    std::vector<int> inputMode_;
    std::optional<std::vector<int>> outputMode_;
    ExtentsMap extents_;
    cudaDataType_t inputElementsType_;
    ngraph::element::Type_t permutationElementsType_;
};

} // namespace CUDAPlugin

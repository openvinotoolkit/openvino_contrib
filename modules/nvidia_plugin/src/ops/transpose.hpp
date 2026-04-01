// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda_operation_base.hpp>
#include <optional>
#include <unordered_map>
#include <vector>

namespace ov {
namespace nvidia_gpu {

class TransposeOp : public OperationCuTensor {
public:
    TransposeOp(const CreationContext& context,
                const std::shared_ptr<ov::Node>& node,
                IndexCollection&& inputIds,
                IndexCollection&& outputIds);

    void Execute(const InferenceRequestContext& context,
                 Inputs inputTensors,
                 Outputs outputTensors,
                 const Workbuffers& workbuffers) const override;

    CudaGraphCompatibility GetCudaGraphCompatibilityImpl() const override;

private:
    using ExtentsMap = std::unordered_map<int, std::int64_t>;

    static std::vector<std::int64_t> extractInputExtents(const ov::Node& node);

    static std::vector<std::int64_t> extractOutputExtents(const ov::Node& node);

    static std::vector<int> extractInputMode(std::size_t numDims);

    static std::vector<std::int64_t> extractInputStrides(const ov::Node& node);

    static std::vector<std::int64_t> extractOutputStrides(const ov::Node& node);

    static ExtentsMap extractExtents(const std::vector<std::int64_t>& input_extents);

    static bool isPermutationTensorSpecified(const ov::Node& node);

    static std::optional<std::vector<int>> tryToExtractPermutation(const ov::Node& node);

    std::vector<int> permutation(const InferenceRequestContext& context, Inputs inputTensors) const;

    ov::element::Type_t extractPermutationElementsType(const ov::Node& node);

    template <typename T>
    static std::vector<int> downloadPermutationVector(const InferenceRequestContext& context,
                                                      CUDA::DevicePointer<const void*>,
                                                      unsigned numDims);

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
    ov::element::Type_t permutationElementsType_;
};

}  // namespace nvidia_gpu
}  // namespace ov

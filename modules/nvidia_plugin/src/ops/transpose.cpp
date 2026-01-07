// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transpose.hpp"

#include <fmt/format.h>

#include <algorithm>
#include <cuda/tensor.hpp>
#include <cuda_operation_registry.hpp>
#include <openvino/core/except.hpp>
#include <openvino/op/constant.hpp>

#include "converters.hpp"
#include "cuda/constant_factory.hpp"

using namespace std::string_literals;

namespace ov {
namespace nvidia_gpu {

inline bool isInputElementsTypeSupported(cudaDataType_t type) {
    switch (type) {
        case CUDA_R_16F:
        case CUDA_R_32F:
        case CUDA_R_16BF:
        case CUDA_R_64F:
            return true;
        default:
            return false;
    }
}

inline bool isPermutationElementsTypeSupported(ov::element::Type_t type) {
    using ov::element::Type_t;
    switch (type) {
        case Type_t::i8:
        case Type_t::i16:
        case Type_t::i32:
        case Type_t::i64:
        case Type_t::u8:
        case Type_t::u16:
        case Type_t::u32:
        case Type_t::u64:
            return true;
        default:
            return false;
    }
}

TransposeOp::TransposeOp(const CreationContext& context,
                         const std::shared_ptr<ov::Node>& node,
                         IndexCollection&& inputIds,
                         IndexCollection&& outputIds)
    : OperationCuTensor(context, node, std::move(inputIds), std::move(outputIds)),
      inputExtents_{extractInputExtents(*node)},
      dimsNumber_{inputExtents_.size()},
      outputExtents_{extractOutputExtents(*node)},
      inputStrides_{extractInputStrides(*node)},
      outputStrides_{extractOutputStrides(*node)},
      inputMode_{extractInputMode(dimsNumber_)},
      outputMode_{tryToExtractPermutation(*node)},
      extents_{extractExtents(inputExtents_)},
      inputElementsType_{convertDataType<cudaDataType_t>(node->input(0).get_element_type())},
      permutationElementsType_{extractPermutationElementsType(*node)} {
    if (!isInputElementsTypeSupported(inputElementsType_)) {
        throw_ov_exception(fmt::format("TransposeOp: unsupported inputElementsType_: {}", toString(inputElementsType_)));
    }
    if (!isPermutationElementsTypeSupported(permutationElementsType_)) {
        throw_ov_exception(fmt::format("TransposeOp: unsupported permutationElementsType_: {}",
                                     ov::element::Type{permutationElementsType_}.get_type_name()));
    }
    inputExtents_.size();
}

void TransposeOp::Execute(const InferenceRequestContext& context,
                          Inputs inputTensors,
                          Outputs outputTensors,
                          const Workbuffers&) const {
    OPENVINO_ASSERT(inputTensors.size() == 1 || inputTensors.size() == 2, "Node name: ", GetName());
    OPENVINO_ASSERT(outputTensors.size() == 1, "Node name: ", GetName());

    cutensorTensorDescriptor_t inputDesc{}, outputDesc{};
    const std::vector<int> outputMode = permutation(context, inputTensors);
    auto& threadContext = context.getThreadContext();

    cutensorInitTensorDescriptor(&threadContext.cuTensorHandle().get(),
                                 &inputDesc,
                                 dimsNumber_,
                                 inputExtents_.data(),
                                 inputStrides_.data(),
                                 inputElementsType_,
                                 CUTENSOR_OP_IDENTITY);

    cutensorInitTensorDescriptor(&threadContext.cuTensorHandle().get(),
                                 &outputDesc,
                                 dimsNumber_,
                                 outputExtents_.data(),
                                 outputStrides_.data(),
                                 inputElementsType_,
                                 CUTENSOR_OP_IDENTITY);

    throwIfError(cutensorPermutation(&threadContext.cuTensorHandle().get(),
                                     &CUDA::NumericConst<CUDA::constants::one>(inputElementsType_),
                                     inputTensors[0].get(),
                                     &inputDesc,
                                     inputMode_.data(),
                                     outputTensors[0].get(),
                                     &outputDesc,
                                     outputMode.data(),
                                     inputElementsType_,
                                     context.getThreadContext().stream().get()));
}

CudaGraphCompatibility TransposeOp::GetCudaGraphCompatibility() const { return CudaGraphCompatibility::FULL; }

std::vector<std::int64_t> TransposeOp::extractInputExtents(const ov::Node& node) {
    std::vector<std::int64_t> result;
    auto inputShape = node.input(0).get_shape();
    result.reserve(inputShape.size());
    for (auto extent : inputShape) result.emplace_back(extent);
    return result;
}

std::vector<std::int64_t> TransposeOp::extractOutputExtents(const ov::Node& node) {
    std::vector<std::int64_t> result;
    auto outputShape = node.output(0).get_shape();
    result.reserve(outputShape.size());
    for (auto extent : outputShape) result.emplace_back(extent);
    return result;
}

std::vector<std::int64_t> TransposeOp::extractInputStrides(const ov::Node& node) {
    std::vector<std::int64_t> result;
    auto inputShape = node.input(0).get_shape();
    result.reserve(inputShape.size());
    const auto numInputShapeElements = inputShape.size();
    for (std::size_t i = 0; i < numInputShapeElements; i++) result.push_back(ov::row_major_stride(inputShape, i));
    return result;
}

TransposeOp::ExtentsMap TransposeOp::extractExtents(const std::vector<std::int64_t>& input_extents) {
    ExtentsMap result;
    const auto numInputExtents = input_extents.size();
    for (std::size_t i = 0; i < numInputExtents; i++) result.emplace(i, input_extents[i]);
    return result;
}

std::vector<int> TransposeOp::extractInputMode(std::size_t numDims) {
    std::vector<int> result;
    for (int i = 0; i < numDims; i++) result.emplace_back(i);
    return result;
}

std::vector<std::int64_t> TransposeOp::extractOutputStrides(const ov::Node& node) {
    std::vector<std::int64_t> result;
    auto outputShape = node.output(0).get_shape();
    result.reserve(outputShape.size());
    const auto numOutputShapeElements = outputShape.size();
    for (std::size_t i = 0; i < numOutputShapeElements; i++) result.push_back(ov::row_major_stride(outputShape, i));
    return result;
}

bool TransposeOp::isPermutationTensorSpecified(const ov::Node& node) {
    const auto numInputs = node.get_input_size();
    OPENVINO_ASSERT(numInputs == 1 || numInputs == 2);
    return numInputs == 2;
}

std::optional<std::vector<int> > TransposeOp::tryToExtractPermutation(const ov::Node& node) {
    if (isPermutationTensorSpecified(node)) {
        auto nodeRawPtr = node.input(1).get_source_output().get_node();
        if (ov::is_type<const ov::op::v0::Constant>(nodeRawPtr)) {
            // Typically permutation vector is small and comes from constant node.
            // That allows to optimize out copying it from device memory in most cases.
            auto constant = dynamic_cast<const ov::op::v0::Constant*>(nodeRawPtr);
            return constant->cast_vector<int>();
        } else {
            return std::nullopt;
        }
    } else {
        auto result = extractInputMode(node.get_input_shape(0).size());
        std::reverse(result.begin(), result.end());
        return result;
    }
}

std::vector<int> TransposeOp::permutation(const InferenceRequestContext& context, Inputs inputTensors) const {
    if (outputMode_.has_value()) {
        return outputMode_.value();
    } else {  // Copies permutation vector from device memory. cuTENSOR API requires it in host memory
        OPENVINO_ASSERT(inputTensors.size() == 2, "Node name: ", GetName());
        using ov::element::Type_t;
        switch (permutationElementsType_) {
            case Type_t::i8:
                return downloadPermutationVector<std::int8_t>(context, inputTensors[1], dimsNumber_);
            case Type_t::i16:
                return downloadPermutationVector<std::int16_t>(context, inputTensors[1], dimsNumber_);
            case Type_t::i32:
                return downloadPermutationVector<std::int32_t>(context, inputTensors[1], dimsNumber_);
            case Type_t::i64:
                return downloadPermutationVector<std::int64_t>(context, inputTensors[1], dimsNumber_);
            case Type_t::u8:
                return downloadPermutationVector<std::uint8_t>(context, inputTensors[1], dimsNumber_);
            case Type_t::u16:
                return downloadPermutationVector<std::uint16_t>(context, inputTensors[1], dimsNumber_);
            case Type_t::u32:
                return downloadPermutationVector<std::uint32_t>(context, inputTensors[1], dimsNumber_);
            case Type_t::u64:
                return downloadPermutationVector<std::uint64_t>(context, inputTensors[1], dimsNumber_);
            default:
                throw_ov_exception("Permutation vector is not of integer type.");
        }
    }
}

ov::element::Type_t TransposeOp::extractPermutationElementsType(const ov::Node& node) {
    OPENVINO_ASSERT(node.get_input_size() > 0 && node.get_input_size() <= 2, "Node name: ", GetName());
    if (node.get_input_size() == 1)
        return ov::element::Type_t::i32;
    else
        return node.get_input_element_type(1);
}

template <typename T>
inline std::vector<int> TransposeOp::downloadPermutationVector(const InferenceRequestContext& context,
                                                               CUDA::DevicePointer<const void*> devicePointer,
                                                               unsigned numDims) {
    std::vector<int> result;
    result.reserve(numDims);
    std::vector<T> perm(numDims);
    context.getThreadContext().stream().download(perm.data(), devicePointer, numDims * sizeof(T));
    context.getThreadContext().stream().synchronize();
    std::copy(perm.begin(), perm.end(), std::back_inserter(result));
    return result;
}

OPERATION_REGISTER(TransposeOp, Transpose);

}  // namespace nvidia_gpu
}  // namespace ov

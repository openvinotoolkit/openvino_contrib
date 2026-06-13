// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transpose.hpp"

#include "kernels/permute_fallback.hpp"

#include <fmt/format.h>

#include <cstdio>
#include <mutex>
#include <cstdlib>

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

    // cuTENSOR does not check the result of its internal kernel launches: on a
    // GPU for which the linked cuTENSOR build has no kernel image (e.g.
    // cuTENSOR 1.x on Ada), cutensorPermutation returns SUCCESS while the
    // output stays unwritten. Probe actual functionality once per process and
    // route Transpose through the built-in fallback kernel if it is broken.
    if (!isCuTensorPermutationFunctional(threadContext)) {
        runPermuteFallback(context, inputTensors[0].get(), outputTensors[0].get(), outputMode);
        return;
    }

    throwIfError(cutensorInitTensorDescriptor(&threadContext.cuTensorHandle().get(),
                                 &inputDesc,
                                 dimsNumber_,
                                 inputExtents_.data(),
                                 inputStrides_.data(),
                                 inputElementsType_,
                                 CUTENSOR_OP_IDENTITY));

    throwIfError(cutensorInitTensorDescriptor(&threadContext.cuTensorHandle().get(),
                                 &outputDesc,
                                 dimsNumber_,
                                 outputExtents_.data(),
                                 outputStrides_.data(),
                                 inputElementsType_,
                                 CUTENSOR_OP_IDENTITY));

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

bool TransposeOp::isCuTensorPermutationFunctional(const ThreadContext& threadContext) {
    static std::once_flag probe_once;
    static bool functional = false;
    std::call_once(probe_once, [&threadContext] {
        functional = [&threadContext]() -> bool {
            cudaStream_t probe_stream = nullptr;
            if (cudaStreamCreateWithFlags(&probe_stream, cudaStreamNonBlocking) != cudaSuccess) {
                return false;
            }
            float* buf = nullptr;
            bool ok = false;
            const float host_src[4] = {1.0f, 2.0f, 3.0f, 4.0f};
            float host_dst[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            do {
                if (cudaMalloc(&buf, 8 * sizeof(float)) != cudaSuccess) {
                    break;
                }
                float* src = buf;
                float* dst = buf + 4;
                if (cudaMemcpyAsync(src, host_src, sizeof(host_src), cudaMemcpyHostToDevice, probe_stream) !=
                    cudaSuccess) {
                    break;
                }
                if (cudaMemsetAsync(dst, 0, 4 * sizeof(float), probe_stream) != cudaSuccess) {
                    break;
                }
                const int64_t extents[2] = {2, 2};
                const int64_t in_strides[2] = {2, 1};
                const int64_t out_strides[2] = {1, 2};  // output laid out transposed
                const int modes[2] = {0, 1};
                cutensorTensorDescriptor_t in_desc{}, out_desc{};
                if (cutensorInitTensorDescriptor(&threadContext.cuTensorHandle().get(), &in_desc, 2, extents,
                                                 in_strides, CUDA_R_32F, CUTENSOR_OP_IDENTITY) !=
                    CUTENSOR_STATUS_SUCCESS) {
                    break;
                }
                if (cutensorInitTensorDescriptor(&threadContext.cuTensorHandle().get(), &out_desc, 2, extents,
                                                 out_strides, CUDA_R_32F, CUTENSOR_OP_IDENTITY) !=
                    CUTENSOR_STATUS_SUCCESS) {
                    break;
                }
                const float one = 1.0f;
                if (cutensorPermutation(&threadContext.cuTensorHandle().get(), &one, src, &in_desc, modes, dst,
                                        &out_desc, modes, CUDA_R_32F, probe_stream) != CUTENSOR_STATUS_SUCCESS) {
                    break;
                }
                if (cudaMemcpyAsync(host_dst, dst, sizeof(host_dst), cudaMemcpyDeviceToHost, probe_stream) !=
                    cudaSuccess) {
                    break;
                }
                if (cudaStreamSynchronize(probe_stream) != cudaSuccess) {
                    break;
                }
                (void)cudaGetLastError();  // clear any launch error swallowed by cuTENSOR
                // dst is the transposed layout of src: expect {1,3,2,4} element order
                ok = host_dst[0] == 1.0f && host_dst[1] == 3.0f && host_dst[2] == 2.0f && host_dst[3] == 4.0f;
            } while (false);
            if (buf) {
                (void)cudaFree(buf);
            }
            (void)cudaStreamDestroy(probe_stream);
            if (!ok) {
                fprintf(stderr,
                        "[NVIDIA plugin] cuTENSOR permutation is not functional on this GPU "
                        "(no kernel image for this architecture?); Transpose will use the built-in "
                        "fallback permutation kernel.\n");
            }
            return ok;
        }();
    });
    return functional;
}

void TransposeOp::runPermuteFallback(const InferenceRequestContext& context,
                                     const void* src,
                                     void* dst,
                                     const std::vector<int>& outputMode) const {
    OPENVINO_ASSERT(dimsNumber_ <= kernel::PermuteFallbackParams::kMaxRank,
                    "Transpose fallback supports up to ",
                    kernel::PermuteFallbackParams::kMaxRank,
                    " dims, got ",
                    dimsNumber_,
                    "; node: ",
                    GetName());
    static std::once_flag warn_once;
    std::call_once(warn_once, [] {
        fprintf(stderr,
                "[NVIDIA plugin] cuTENSOR permutation kernels are unavailable on this GPU; "
                "using the built-in fallback permutation kernel for Transpose.\n");
    });
    kernel::PermuteFallbackParams params{};
    params.rank = static_cast<int>(dimsNumber_);
    params.num_elements = 1;
    for (size_t k = 0; k < dimsNumber_; ++k) {
        params.out_extents[k] = outputExtents_[k];
        params.out_strides[k] = outputStrides_[k];
        params.in_strides_permuted[k] = inputStrides_[outputMode[k]];
        params.num_elements *= static_cast<size_t>(outputExtents_[k]);
    }
    size_t elem_size = 0;
    switch (inputElementsType_) {
        case CUDA_R_8I:
        case CUDA_R_8U:
            elem_size = 1;
            break;
        case CUDA_R_16F:
#if CUDART_VERSION >= 11000
        case CUDA_R_16BF:
#endif
        case CUDA_R_16I:
        case CUDA_R_16U:
            elem_size = 2;
            break;
        case CUDA_R_32F:
        case CUDA_R_32I:
        case CUDA_R_32U:
            elem_size = 4;
            break;
        case CUDA_R_64F:
        case CUDA_R_64I:
        case CUDA_R_64U:
            elem_size = 8;
            break;
        default:
            throw_ov_exception(fmt::format("Transpose fallback: unsupported element type {}", toString(inputElementsType_)));
    }
    kernel::permute_fallback(context.getThreadContext().stream().get(), params, elem_size, src, dst);
}

CudaGraphCompatibility TransposeOp::GetCudaGraphCompatibilityImpl() const { return CudaGraphCompatibility::FULL; }

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

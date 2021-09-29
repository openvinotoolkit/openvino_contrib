// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "details/cuda_ie_api_import_fix.hpp"
// ^^ must come before any other ie includes which use
// INFERENCE_ENGINE_DEPRECATED
#include "details/cuda_ngraph_import_fix.hpp"
// ^^ must come before any other ngraph includes which use
// NGRAPH_DEPRECATED
#include <fmt/format.h>

#include <cuda_operation_registry.hpp>

#include "ngraph/axis_set.hpp"
#include "ngraph/op/constant.hpp"
#include "strided_slice.hpp"

namespace CUDAPlugin {
namespace kernel {

#ifdef CUDA_KERNEL_PRINT_LOG
template <typename T>
static __device__ int INT(T t) {
    return static_cast<int>(t);
}
#endif

template <typename T>
static __global__ void reverse(const size_t maxSize,
                               const size_t chunksNumber,
                               const size_t notReversedChunkSize,
                               T* buffer) {
    /*
    Let's assume there is a matrix with {2, 3, 9} shape and the axis #1 has to be swapped,
    then chunksNumber and notReversedChunkSize have following values
    [
      |*                chunkNumber = 3                                  *|
      |*  notReversedChunkSize = 8 elements  *|
    [ [ 100, 200, 300, 400, 500, 600, 700, 800],  [2, ...], [ 3 .....] ]
    [ [.............                          ],  [...,..], [ .......] ]
    ]
    And maxSize = 54  (2 * 3 * 9)
    */

    const auto idx = static_cast<size_t>(blockIdx.x * blockDim.x + threadIdx.x);
    const size_t sectionSize = chunksNumber * notReversedChunkSize;
    const size_t threadsPerSection = sectionSize >> 1;
    const size_t startPos = (idx / threadsPerSection) * sectionSize;

    const auto maxThreadIdx = maxSize >> 1;
    if (idx >= maxThreadIdx) {
#ifdef CUDA_KERNEL_PRINT_LOG
        printf("reverse #0 threadIdx = %d exceeded array size (maxThreadIdx = %d) \n", INT(idx), INT(maxThreadIdx));
#endif
        return;
    }
    const size_t idxFromStartPos = idx % threadsPerSection;
    const size_t chunkIdA = idxFromStartPos / notReversedChunkSize;
    const size_t indexWithinChunk = idxFromStartPos % notReversedChunkSize;
    const size_t chunkIdB = chunksNumber - chunkIdA - 1;
    const size_t offsetA = startPos + idxFromStartPos;
    const size_t offsetB = startPos + chunkIdB * notReversedChunkSize + indexWithinChunk;
#ifdef CUDA_KERNEL_PRINT_LOG
    printf(
        "reverse #1 threadIdx = %d chunksNumber = %d startPos = %d chunkIdA = %d chunkIdB = %d indexWithinChunk = %d"
        " offsetA = %d offsetB = %d\n",
        INT(idx),
        INT(chunksNumber),
        INT(startPos),
        INT(chunkIdA),
        INT(chunkIdB),
        INT(indexWithinChunk),
        INT(offsetA),
        INT(offsetB));
#endif
    T tempVal = buffer[offsetA];
    buffer[offsetA] = buffer[offsetB];
    buffer[offsetB] = tempVal;
}

template <typename T>
static __global__ void strideSlice(const size_t shapeSize,
                                   const int64_t* srcMatrixSizes,
                                   const T* src,
                                   const int64_t* begin,
                                   const int64_t* end,
                                   const int64_t* stride,
                                   const int64_t* dstMatrixSizes,
                                   T* dst) {
    const int64_t dstIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (dstIdx >= dstMatrixSizes[0]) {
        return;
    }
#ifdef CUDA_KERNEL_PRINT_LOG
    printf("#0 dstIdx = %d shapeSize = %d blockIdx.x = %d blockDim.x = %d threadIdx.x = %d\n",
           INT(dstIdx),
           INT(shapeSize),
           INT(blockIdx.x),
           INT(blockDim.x),
           INT(threadIdx.x));
#endif
    int64_t i = dstIdx;
    int64_t srcOffset = 0;
    int64_t coordIdx = 0;
    int64_t dstCoord = 0;
    for (size_t idx = 1; idx < shapeSize; ++idx, ++coordIdx) {
        dstCoord = i / dstMatrixSizes[idx];
        i -= dstCoord * dstMatrixSizes[idx];
        srcOffset = srcOffset + srcMatrixSizes[idx] * (begin[coordIdx] + dstCoord * stride[coordIdx]);
#ifdef CUDA_KERNEL_PRINT_LOG
        printf(
            "#1 dstIdx = %d # srcMatrixSizes[idx] = %d begin[coordIdx] = %d srcOffset = %d curStride = %d"
            " relDstCoord = %d idx= %u \n",
            INT(dstIdx),
            INT(srcMatrixSizes[idx]),
            INT(begin[coordIdx]),
            INT(srcOffset),
            INT(stride[coordIdx]),
            INT(dstCoord),
            INT(idx));
#endif
    }
    dstCoord = i;

    srcOffset = srcOffset + begin[coordIdx] + dstCoord * stride[coordIdx];
#ifdef CUDA_KERNEL_PRINT_LOG
    printf("#3_0 dstIdx = %d begin[coordIdx] = %d curStride=%d i=%d\n",
           INT(dstIdx),
           INT(begin[coordIdx]),
           INT(stride[coordIdx]),
           INT(i));
    printf("#3_1 dstIdx = %d, srcOffset = %d\n", INT(dstIdx), INT(srcOffset));
#endif
    dst[dstIdx] = src[srcOffset];
}

}  // namespace kernel

namespace {
ngraph::AxisSet convert_mask_to_axis_set(const std::vector<int64_t>& mask) {
    ngraph::AxisSet axis_set;
    for (size_t i = 0; i < mask.size(); ++i) {
        if (mask[i] == 1) {
            axis_set.insert(i);
        }
    }
    return axis_set;
}

void calcMatrixSizes(const ngraph::Shape& shape, std::vector<int64_t>& matrix) {
    size_t prev_shape_size = 1;

    for (size_t src_shape_idx = shape.size(); src_shape_idx > 0; --src_shape_idx) {
        prev_shape_size = shape[src_shape_idx - 1] * prev_shape_size;
        matrix[src_shape_idx - 1] = prev_shape_size;
    }
}

template <typename T>
auto size_bytes(const std::vector<T>& v) noexcept {
    return sizeof(T) * v.size();
}

}  // namespace

StridedSliceOp::StridedSliceOp(const CreationContext& context,
                               const NodeOp& stridedSliceOp,
                               IndexCollection&& inputIds,
                               IndexCollection&& outputIds)
    : OperationBase(context, stridedSliceOp, std::move(inputIds), std::move(outputIds)),
      element_type_{stridedSliceOp.get_input_element_type(0)} {
    const auto begin_const = getNodeConstantValues(stridedSliceOp.get_input_node_ptr(1));
    const auto end_const = getNodeConstantValues(stridedSliceOp.get_input_node_ptr(2));
    const auto stride_const = getNodeConstantValues(stridedSliceOp.get_input_node_ptr(3));
    slice_plan = ngraph::make_slice_plan(stridedSliceOp.get_input_shape(0),
                                         begin_const,
                                         end_const,
                                         stride_const,
                                         convert_mask_to_axis_set(stridedSliceOp.get_begin_mask()),
                                         convert_mask_to_axis_set(stridedSliceOp.get_end_mask()),
                                         convert_mask_to_axis_set(stridedSliceOp.get_new_axis_mask()),
                                         convert_mask_to_axis_set(stridedSliceOp.get_shrink_axis_mask()),
                                         convert_mask_to_axis_set(stridedSliceOp.get_ellipsis_mask()));
    src_matrix_sizes = std::vector<int64_t>(stridedSliceOp.get_input_shape(0).size(), 0);
    dst_matrix_sizes = std::vector<int64_t>(slice_plan.reshape_in_shape.size(), 0);
    calcMatrixSizes(stridedSliceOp.get_input_shape(0), src_matrix_sizes);
    calcMatrixSizes(slice_plan.reshape_in_shape, dst_matrix_sizes);

    const auto& prop = context.device().props();
    max_threads_per_block_ = prop.maxThreadsPerBlock;
    blocks_number_ = 1 + dst_matrix_sizes[0] / max_threads_per_block_;
    threads_per_block_ = (blocks_number_ == 1) ? dst_matrix_sizes[0] : max_threads_per_block_;
}

void StridedSliceOp::Execute(const InferenceRequestContext& context,
                             Inputs inputs,
                             Outputs outputs,
                             const Workbuffers& workbuffers) const {
    switch (element_type_) {
        case ngraph::element::Type_t::f32:
            return callKernels<float>(context, inputs, outputs, workbuffers);
        case ngraph::element::Type_t::i32:
            return callKernels<int32_t>(context, inputs, outputs, workbuffers);
        case ngraph::element::Type_t::f16:
            return callKernels<__half>(context, inputs, outputs, workbuffers);
        case ngraph::element::Type_t::i16:
            return callKernels<int16_t>(context, inputs, outputs, workbuffers);
        case ngraph::element::Type_t::i8:
            return callKernels<int8_t>(context, inputs, outputs, workbuffers);
        case ngraph::element::Type_t::u8:
            return callKernels<uint8_t>(context, inputs, outputs, workbuffers);
        default:
            CUDAPlugin::throwIEException(
                fmt::format("Input element type = {} is not supported by StridedSlice operation !!",
                            ngraph::element::Type(element_type_).get_type_name()));
    }
}

WorkbufferRequest StridedSliceOp::GetWorkBufferRequest() const {
    return {{size_bytes(src_matrix_sizes),
             size_bytes(dst_matrix_sizes),
             size_bytes(slice_plan.begins),
             size_bytes(slice_plan.ends),
             size_bytes(slice_plan.strides)},
            {}};
}

void StridedSliceOp::InitSharedImmutableWorkbuffers(const Buffers& buffers) {
    uploadDataToWorkbuffer(buffers[0], src_matrix_sizes);
    uploadDataToWorkbuffer(buffers[1], dst_matrix_sizes);

    uploadDataToWorkbuffer(buffers[2], slice_plan.begins);
    uploadDataToWorkbuffer(buffers[3], slice_plan.ends);
    uploadDataToWorkbuffer(buffers[4], slice_plan.strides);
}

template <typename T>
void StridedSliceOp::callKernels(const InferenceRequestContext& context,
                                 Inputs inputs,
                                 Outputs outputs,
                                 const Workbuffers& workbuffers) const {
    callStridedSliceKernel<T>(context, inputs, outputs, workbuffers);
    callReverseAxesKernel<T>(context, outputs);
}

template <typename T>
void StridedSliceOp::callStridedSliceKernel(const InferenceRequestContext& context,
                                            Inputs inputs,
                                            Outputs outputs,
                                            const Workbuffers& workbuffers) const {
    auto& threadContext = context.getThreadContext();
    auto& stream = threadContext.stream();

    stream.run(blocks_number_,
               threads_per_block_,
               kernel::strideSlice<T>,
               src_matrix_sizes.size(),
               static_cast<const int64_t*>(workbuffers.immutable_buffers[0].get()),
               static_cast<const T*>(inputs[0].get()),
               static_cast<const int64_t*>(workbuffers.immutable_buffers[2].get()),
               static_cast<const int64_t*>(workbuffers.immutable_buffers[3].get()),
               static_cast<const int64_t*>(workbuffers.immutable_buffers[4].get()),
               static_cast<const int64_t*>(workbuffers.immutable_buffers[1].get()),
               static_cast<T*>(outputs[0].get()));
}

template <typename T>
void StridedSliceOp::callReverseAxesKernel(const InferenceRequestContext& context, Outputs outputs) const {
    callReverseAxesKernel<T>(
        context, slice_plan.reshape_out_shape, dst_matrix_sizes, slice_plan.reverse_axes, outputs[0]);
}

template <typename T>
void StridedSliceOp::callReverseAxesKernel(const InferenceRequestContext& context,
                                           const std::vector<size_t>& matrixShapes,
                                           const std::vector<int64_t>& matrixSizes,
                                           const ngraph::AxisSet& reverseAxes,
                                           CUDA::DevicePointer<void*> buffer) const {
    for (auto axisIt = reverseAxes.rbegin(); axisIt != reverseAxes.rend(); ++axisIt) {
        const auto chunksNumber =
            *axisIt < matrixSizes.size() - 1 ? matrixSizes[*axisIt] / matrixSizes[*axisIt + 1] : matrixSizes[*axisIt];
        const auto notReversedChunkSize = *axisIt < matrixSizes.size() - 1 ? matrixSizes[*axisIt + 1] : 1;
        const auto threadsNumber = matrixSizes[0] / 2;

        const unsigned maxBlockSize = max_threads_per_block_;
        const unsigned numBlocks = 1 + threadsNumber / maxBlockSize;
        unsigned threadsPerBlock = (numBlocks == 1) ? threadsNumber : maxBlockSize;

        auto& stream = context.getThreadContext().stream();
        stream.run(blocks_number_,
                   threadsPerBlock,
                   kernel::reverse<T>,
                   static_cast<size_t>(matrixSizes[0]),
                   static_cast<size_t>(chunksNumber),
                   static_cast<size_t>(notReversedChunkSize),
                   static_cast<T*>(buffer.get()));
    }
}

void StridedSliceOp::uploadDataToWorkbuffer(CUDA::DevicePointer<void*> buffer, const std::vector<int64_t>& data) {
    auto& stream = CUDA::DefaultStream::stream();
    stream.upload(buffer, data.data(), size_bytes(data));
}

std::vector<int64_t> StridedSliceOp::getNodeConstantValues(const ngraph::Node* node) const {
    auto constant = dynamic_cast<const ngraph::op::v0::Constant*>(node);
    assert(constant);
    auto begin = reinterpret_cast<const int64_t*>(constant->get_data_ptr());
    return std::vector<int64_t>(begin, begin + shape_size(constant->get_shape()));
}

OPERATION_REGISTER(StridedSliceOp, StridedSlice);

}  // namespace CUDAPlugin

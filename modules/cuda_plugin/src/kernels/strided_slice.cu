// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <fmt/format.h>

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

StridedSliceKernelOp::StridedSliceKernelOp(const std::vector<int64_t> src_matrix_sizes,
                                           const std::vector<int64_t> dst_matrix_sizes,
                                           const std::set<size_t> reverse_axes,
                                           const unsigned max_threads_per_block,
                                           const unsigned blocks_number,
                                           const unsigned threads_per_block,
                                           const ngraph::element::Type_t element_type)
    : src_matrix_sizes_{src_matrix_sizes},
      dst_matrix_sizes_{dst_matrix_sizes},
      reverse_axes_{reverse_axes},
      max_threads_per_block_{max_threads_per_block},
      blocks_number_{blocks_number},
      threads_per_block_{threads_per_block},
      element_type_{element_type} {}

void StridedSliceKernelOp::operator()(const cudaStream_t stream,
                                      const int64_t* src_matrix_sizes,
                                      const void* src,
                                      const int64_t* begin,
                                      const int64_t* end,
                                      const int64_t* stride,
                                      const int64_t* dst_matrix_sizes,
                                      void* dst) const {
    switch (element_type_) {
        case ngraph::element::Type_t::f32:
            return callKernels<float>(stream, src_matrix_sizes, src, begin, end, stride, dst_matrix_sizes, dst);
        case ngraph::element::Type_t::i32:
            return callKernels<int32_t>(stream, src_matrix_sizes, src, begin, end, stride, dst_matrix_sizes, dst);
        case ngraph::element::Type_t::f16:
            return callKernels<__half>(stream, src_matrix_sizes, src, begin, end, stride, dst_matrix_sizes, dst);
        case ngraph::element::Type_t::i16:
            return callKernels<int16_t>(stream, src_matrix_sizes, src, begin, end, stride, dst_matrix_sizes, dst);
        case ngraph::element::Type_t::i8:
            return callKernels<int8_t>(stream, src_matrix_sizes, src, begin, end, stride, dst_matrix_sizes, dst);
        case ngraph::element::Type_t::u8:
            return callKernels<uint8_t>(stream, src_matrix_sizes, src, begin, end, stride, dst_matrix_sizes, dst);
        default:
            throwIEException(fmt::format("Input element type = {} is not supported by StridedSlice operation !!",
                                         ngraph::element::Type(element_type_).get_type_name()));
    }
}

template <typename T>
void StridedSliceKernelOp::callKernels(const cudaStream_t stream,
                                       const int64_t* src_matrix_sizes,
                                       const void* src,
                                       const int64_t* begin,
                                       const int64_t* end,
                                       const int64_t* stride,
                                       const int64_t* dst_matrix_sizes,
                                       void* dst) const {
    callStridedSliceKernel<T>(stream, src_matrix_sizes, src, begin, end, stride, dst_matrix_sizes, dst);
    callReverseAxesKernel<T>(stream, dst);
}

template <typename T>
void StridedSliceKernelOp::callStridedSliceKernel(const cudaStream_t stream,
                                                  const int64_t* src_matrix_sizes,
                                                  const void* src,
                                                  const int64_t* begin,
                                                  const int64_t* end,
                                                  const int64_t* stride,
                                                  const int64_t* dst_matrix_sizes,
                                                  void* dst) const {
    strideSlice<T><<<blocks_number_, threads_per_block_, 0, stream>>>(src_matrix_sizes_.size(),
                                                                      src_matrix_sizes,
                                                                      static_cast<const T*>(src),
                                                                      begin,
                                                                      end,
                                                                      stride,
                                                                      dst_matrix_sizes,
                                                                      static_cast<T*>(dst));
}

template <typename T>
void StridedSliceKernelOp::callReverseAxesKernel(const cudaStream_t stream, void* dst) const {
    for (auto axisIt = reverse_axes_.rbegin(); axisIt != reverse_axes_.rend(); ++axisIt) {
        const auto chunksNumber = *axisIt < dst_matrix_sizes_.size() - 1
                                      ? dst_matrix_sizes_[*axisIt] / dst_matrix_sizes_[*axisIt + 1]
                                      : dst_matrix_sizes_[*axisIt];
        const auto notReversedChunkSize = *axisIt < dst_matrix_sizes_.size() - 1 ? dst_matrix_sizes_[*axisIt + 1] : 1;
        const auto threadsNumber = dst_matrix_sizes_[0] / 2;

        const unsigned maxBlockSize = max_threads_per_block_;
        const unsigned numBlocks = 1 + threadsNumber / maxBlockSize;
        unsigned threadsPerBlock = (numBlocks == 1) ? threadsNumber : maxBlockSize;

        reverse<T><<<numBlocks, threadsPerBlock, 0, stream>>>(static_cast<size_t>(dst_matrix_sizes_[0]),
                                                              static_cast<size_t>(chunksNumber),
                                                              static_cast<size_t>(notReversedChunkSize),
                                                              static_cast<T*>(dst));
    }
}

}  // namespace kernel

}  // namespace CUDAPlugin

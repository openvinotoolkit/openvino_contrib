// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <fmt/format.h>

#include "details/type_validator.hpp"
#include "strided_slice.hpp"

namespace ov {
namespace nvidia_gpu {
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

template <typename T, typename T_INT>
static __global__ void strideSlice(const size_t shapeSize,
                                   const T_INT* srcMatrixSizes,
                                   const T* src,
                                   const T_INT* begin,
                                   const T_INT* end,
                                   const T_INT* stride,
                                   const T_INT* dstMatrixSizes,
                                   T* dst) {
    const T_INT dstIdx = blockIdx.x * blockDim.x + threadIdx.x;
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
    T_INT i = dstIdx;
    T_INT srcOffset = 0;
    T_INT coordIdx = 0;
    T_INT dstCoord = 0;
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

template <typename T_INT>
StridedSliceKernelOp<T_INT>::StridedSliceKernelOp(const std::vector<T_INT> src_matrix_sizes,
                                                  const std::vector<T_INT> dst_matrix_sizes,
                                                  const std::set<size_t> reverse_axes,
                                                  const unsigned max_threads_per_block,
                                                  const unsigned blocks_number,
                                                  const unsigned threads_per_block,
                                                  const Type_t element_type,
                                                  const Type_t element_type_integer)
    : src_matrix_sizes_{src_matrix_sizes},
      dst_matrix_sizes_{dst_matrix_sizes},
      reverse_axes_{reverse_axes},
      max_threads_per_block_{max_threads_per_block},
      blocks_number_{blocks_number},
      threads_per_block_{threads_per_block},
      element_type_{element_type},
      element_type_integer_{element_type_integer} {
    using StridedSliceElementTypesSwitch =
        ElementTypesSwitch<Type_t::f32, Type_t::i32, Type_t::f16, Type_t::i16, Type_t::i8, Type_t::u8>;
    TypeValidator<StridedSliceElementTypesSwitch>::check(element_type_);
    using StridedSliceIntegerElementTypesSwitch = ElementTypesSwitch<Type_t::i32, Type_t::i64>;
    TypeValidator<StridedSliceIntegerElementTypesSwitch>::check(element_type_integer_);
}

template <typename T_INT>
void StridedSliceKernelOp<T_INT>::operator()(const cudaStream_t stream,
                                             const T_INT* src_matrix_sizes,
                                             const void* src,
                                             const T_INT* begin,
                                             const T_INT* end,
                                             const T_INT* stride,
                                             const T_INT* dst_matrix_sizes,
                                             void* dst) const {
    switch (element_type_) {
        case Type_t::f32:
            return callKernels<float>(stream, src_matrix_sizes, src, begin, end, stride, dst_matrix_sizes, dst);
        case Type_t::i32:
            return callKernels<int32_t>(stream, src_matrix_sizes, src, begin, end, stride, dst_matrix_sizes, dst);
        case Type_t::f16:
            return callKernels<__half>(stream, src_matrix_sizes, src, begin, end, stride, dst_matrix_sizes, dst);
        case Type_t::i16:
            return callKernels<int16_t>(stream, src_matrix_sizes, src, begin, end, stride, dst_matrix_sizes, dst);
        case Type_t::i8:
            return callKernels<int8_t>(stream, src_matrix_sizes, src, begin, end, stride, dst_matrix_sizes, dst);
        case Type_t::u8:
            return callKernels<uint8_t>(stream, src_matrix_sizes, src, begin, end, stride, dst_matrix_sizes, dst);
        default:
            throw_ov_exception(fmt::format("Input element type = {} is not supported by StridedSlice operation !!",
                                         element_type_));
    }
}

template <typename T_INT>
template <typename T>
void StridedSliceKernelOp<T_INT>::callKernels(const cudaStream_t stream,
                                              const T_INT* src_matrix_sizes,
                                              const void* src,
                                              const T_INT* begin,
                                              const T_INT* end,
                                              const T_INT* stride,
                                              const T_INT* dst_matrix_sizes,
                                              void* dst) const {
    callStridedSliceKernel<T>(stream, src_matrix_sizes, src, begin, end, stride, dst_matrix_sizes, dst);
    callReverseAxesKernel<T>(stream, dst);
}

template <typename T_INT>
template <typename T>
void StridedSliceKernelOp<T_INT>::callStridedSliceKernel(const cudaStream_t stream,
                                                         const T_INT* src_matrix_sizes,
                                                         const void* src,
                                                         const T_INT* begin,
                                                         const T_INT* end,
                                                         const T_INT* stride,
                                                         const T_INT* dst_matrix_sizes,
                                                         void* dst) const {
    strideSlice<T, T_INT><<<blocks_number_, threads_per_block_, 0, stream>>>(src_matrix_sizes_.size(),
                                                                             static_cast<const T_INT*>(src_matrix_sizes),
                                                                             static_cast<const T*>(src),
                                                                             static_cast<const T_INT*>(begin),
                                                                             static_cast<const T_INT*>(end),
                                                                             static_cast<const T_INT*>(stride),
                                                                             static_cast<const T_INT*>(dst_matrix_sizes),
                                                                             static_cast<T*>(dst));
            }

template <typename T_INT>
template <typename T>
void StridedSliceKernelOp<T_INT>::callReverseAxesKernel(const cudaStream_t stream, void* dst) const {
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

}  // namespace nvidia_gpu
}  // namespace ov

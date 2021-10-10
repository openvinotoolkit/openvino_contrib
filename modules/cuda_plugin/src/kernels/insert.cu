// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cuda.h>
#if CUDA_VERSION >= 11000
#include <cuda_bf16.h>
#endif
#include <cuda_fp16.h>
#include <fmt/format.h>

#include <error.hpp>
#include <gsl/gsl_assert>

#include "insert.hpp"
#include "tensor_helpers.hpp"

namespace CUDAPlugin {
namespace kernel {

template <typename T>
static __global__ void insert_part(const Insert::Props* props, const size_t start, const T* x, T* y) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t old_rank = rank(props->old_shape);
    const size_t new_rank = rank(props->new_shape);
    assert(old_rank == new_rank);
    Shape<size_t, 5> originalIndexes{};
    shape_indices(props->old_shape, i, originalIndexes);
    Shape<size_t, 5> joinIndexes{};
    memcpy(joinIndexes, originalIndexes, sizeof(originalIndexes));
    joinIndexes[props->axe] = start + joinIndexes[props->axe];
    const size_t flatInputAddress = flat_address(props->new_shape, joinIndexes);
    y[flatInputAddress] = x[i];
}

Insert::Insert(const Type_t element_type, const Props& props, const size_t max_threads_per_block)
    : element_type_{element_type}, props_{props} {
    std::tie(num_blocks_, threads_per_block_) =
        calculateElementwiseGrid(shape_size(props.old_shape), max_threads_per_block);
}

void Insert::operator()(const cudaStream_t stream, const void* src, void* dst, const size_t start) const {
    switch (element_type_) {
        case Type_t::boolean:
            return call<bool>(stream, src, dst, start);
#if CUDA_VERSION >= 11000
        case Type_t::bf16:
            return call<__nv_bfloat16>(stream, src, dst, start);
#endif
        case Type_t::f16:
            return call<__half>(stream, src, dst, start);
        case Type_t::f32:
            return call<float>(stream, src, dst, start);
        case Type_t::f64:
            return call<double>(stream, src, dst, start);
        case Type_t::i8:
            return call<int8_t>(stream, src, dst, start);
        case Type_t::i16:
            return call<int16_t>(stream, src, dst, start);
        case Type_t::i32:
            return call<int32_t>(stream, src, dst, start);
        case Type_t::i64:
            return call<int64_t>(stream, src, dst, start);
        case Type_t::u8:
            return call<uint8_t>(stream, src, dst, start);
        case Type_t::u16:
            return call<uint16_t>(stream, src, dst, start);
        case Type_t::u32:
            return call<uint32_t>(stream, src, dst, start);
        case Type_t::u64:
            return call<uint64_t>(stream, src, dst, start);
        default:
            throwIEException(fmt::format("Input element type = {} is not supported by Split operation !!",
                                         static_cast<Type_t>(element_type_)));
    }
}

template <typename T>
void Insert::call(const cudaStream_t stream, const void* src, void* dst, const size_t start) const {
    Expects(props_ptr_);
    insert_part<T><<<num_blocks_, threads_per_block_, 0, stream>>>(
        static_cast<const Props*>(props_ptr_), start, static_cast<const T*>(src), static_cast<T*>(dst));
}

}  // namespace kernel
}  // namespace CUDAPlugin

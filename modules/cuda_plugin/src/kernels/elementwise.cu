// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cuda.h>
#include <fmt/format.h>

#include <error.hpp>
#include <gsl/gsl_assert>

#include "elementtypeswitch.hpp"
#include "elementwise.hpp"
#include "switch.hpp"
#include "tensor_helpers.hpp"

namespace CUDAPlugin {
namespace kernel {

template <typename T, typename OP>
static __global__ void element_wise(size_t num_elements, const T* in0, const T* in1, T* out) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_elements) {
        out[i] = OP::op(in0[i], in1[i]);
    }
}

template <typename T, Elementwise::Op_t>
struct OpImpl;
template <typename T>
struct OpImpl<T, Elementwise::Op_t::add> {
    __device__ static inline T op(T in0, T in1) { return in0 + in1; }
};
template <typename T>
struct OpImpl<T, Elementwise::Op_t::mul> {
    __device__ static inline T op(T in0, T in1) { return in0 * in1; }
};

template <typename T>
struct OpTypeSwitch {
    size_t max_threads_per_block_;
    using Op_t = Elementwise::Op_t;
    static constexpr std::integer_sequence<Op_t, Op_t::add, Op_t::mul> op_type_identifiers{};

    void operator()(
        Op_t op, cudaStream_t stream, size_t num_elements, const void* in0, const void* in1, void* out) const {
        templateSwitch(op_type_identifiers, op, *this, stream, num_elements, in0, in1, out);
    }

    template <Op_t OP>
    constexpr void case_(
        cudaStream_t stream, size_t num_elements, const void* in0, const void* in1, void* out) const noexcept {
        size_t num_blocks{}, threads_per_block{};
        std::tie(num_blocks, threads_per_block) = calculateElementwiseGrid(num_elements, max_threads_per_block_);
        element_wise<T, OpImpl<T, OP>><<<num_blocks, threads_per_block, 0, stream>>>(
            num_elements, static_cast<const T*>(in0), static_cast<const T*>(in1), static_cast<T*>(out));
    }

    constexpr void default_(
        Op_t t, cudaStream_t stream, size_t num_elements, const void* in0, const void* in1, void* out) const noexcept {
        throwIEException(fmt::format("Operation type = {} is not supported.", t));
    }
};

struct ElementwiseTypeSwitch {
    using Op_t = Elementwise::Op_t;
    Op_t op_type_;
    size_t max_threads_per_block_;

    template <typename T>
    constexpr void case_(
        cudaStream_t stream, size_t num_elements, const void* in0, const void* in1, void* out) const noexcept {
        OpTypeSwitch<T> switchObj{max_threads_per_block_};
        switchObj(op_type_, stream, num_elements, in0, in1, out);
    }
    void default_(Type_t t, cudaStream_t stream, size_t num_elements, const void* in0, const void* in1, void* out)
        const noexcept {
        throwIEException(fmt::format("Element type = {} is not supported.", t));
    }
};

Elementwise::Elementwise(Op_t op_type, Type_t element_type, size_t num_elements, size_t max_threads_per_block)
    : op_type_{op_type},
      element_type_{element_type},
      num_elements_{num_elements},
      max_threads_per_block_{max_threads_per_block} {}

void Elementwise::operator()(cudaStream_t stream, const void* in0, const void* in1, void* out) const {
    using SupportedElementTypes =
        ElementTypesSwitch<Type_t::i16, Type_t::i32, Type_t::i64, Type_t::u8, Type_t::u16, Type_t::u32, Type_t::u64>;
    ElementwiseTypeSwitch switchObj{op_type_, max_threads_per_block_};
    SupportedElementTypes::switch_(element_type_, switchObj, stream, num_elements_, in0, in1, out);
}

}  // namespace kernel
}  // namespace CUDAPlugin

// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cuda.h>
#include <fmt/format.h>

#include <gsl/gsl_assert>

#include "elementtypeswitch.hpp"
#include "elementwise.hpp"
#include "switch.hpp"
#include "tensor_helpers.hpp"

namespace CUDAPlugin {
namespace kernel {

template <typename T, typename OP>
static __global__ void element_wise(const T* in0, const T* in1, T* out, size_t num_elements) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_elements) {
        out[i] = OP::op(in0[i], in1[i]);
    }
}

template <typename T, typename OP>
static __global__ void element_wise_broadcasting(
    const T* in, size_t in_num_elements, const T* broadcasted, size_t broadcasted_num_elements, T* out) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < in_num_elements) {
        out[i] = OP::op(in[i], broadcasted[i % broadcasted_num_elements]);
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
    static constexpr std::integer_sequence< int, static_cast<int>(Op_t::add), static_cast<int>(Op_t::mul)> op_type_identifiers{};

    void operator()(Op_t op,
                    cudaStream_t stream,
                    const void* in0,
                    size_t in0_num_elements,
                    const void* in1,
                    size_t in1_num_elements,
                    void* out) const {
        templateSwitch(op_type_identifiers, op, *this, stream, in0, in0_num_elements, in1, in1_num_elements, out);
    }

    template <Op_t OP>
    constexpr void case_(cudaStream_t stream,
                         const void* in0,
                         size_t in0_num_elements,
                         const void* in1,
                         size_t in1_num_elements,
                         void* out) const noexcept {
        if (in0_num_elements == in1_num_elements) {
            size_t num_blocks{}, threads_per_block{};
            std::tie(num_blocks, threads_per_block) = 
                calculateElementwiseGrid(in0_num_elements, max_threads_per_block_);
            element_wise<T, OpImpl<T, OP>><<<num_blocks, threads_per_block, 0, stream>>>(
                static_cast<const T*>(in0), static_cast<const T*>(in1), static_cast<T*>(out), in0_num_elements);
        } else if (in0_num_elements < in1_num_elements) {
            Expects(in1_num_elements % in0_num_elements == 0);
            size_t num_blocks{}, threads_per_block{};
            std::tie(num_blocks, threads_per_block) =
                calculateElementwiseGrid(in1_num_elements, max_threads_per_block_);
            element_wise_broadcasting<T, OpImpl<T, OP>>
                <<<num_blocks, threads_per_block, 0, stream>>>(static_cast<const T*>(in1),
                                                               in1_num_elements,
                                                               static_cast<const T*>(in0),
                                                               in0_num_elements,
                                                               static_cast<T*>(out));
        } else {
            Expects(in0_num_elements % in1_num_elements == 0);
            size_t num_blocks{}, threads_per_block{};
            std::tie(num_blocks, threads_per_block) =
                calculateElementwiseGrid(in0_num_elements, max_threads_per_block_);
            element_wise_broadcasting<T, OpImpl<T, OP>>
                <<<num_blocks, threads_per_block, 0, stream>>>(static_cast<const T*>(in0),
                                                               in0_num_elements,
                                                               static_cast<const T*>(in1),
                                                               in1_num_elements,
                                                               static_cast<T*>(out));
        }
    }

    constexpr void default_(Op_t t, cudaStream_t, const void*, size_t, const void*, size_t, void*) const noexcept {
        throwIEException(fmt::format("Operation type = {} is not supported.", t));
    }
};

struct ElementwiseTypeSwitch {
    using Op_t = Elementwise::Op_t;
    Op_t op_type_;
    size_t max_threads_per_block_;

    template <typename T>
    constexpr void case_(cudaStream_t stream,
                         const void* in0,
                         size_t in0_num_elements,
                         const void* in1,
                         size_t in1_num_elements,
                         void* out) const noexcept {
        OpTypeSwitch<T> switchObj{max_threads_per_block_};
        switchObj(op_type_, stream, in0, in0_num_elements, in1, in1_num_elements, out);
    }
    template <typename T>
    void default_(T t, cudaStream_t, const void*, size_t, const void*, size_t, void*) const noexcept {
        throwIEException(fmt::format("Element type = {} is not supported.", t));
    }
};

Elementwise::Elementwise(Op_t op_type, Type_t element_type, size_t max_threads_per_block)
    : op_type_{op_type}, element_type_{element_type}, max_threads_per_block_{max_threads_per_block} {}

void Elementwise::operator()(cudaStream_t stream,
                             const void* in0,
                             size_t in0_num_elements,
                             const void* in1,
                             size_t in1_num_elements,
                             void* out) const {
    using SupportedElementTypes =
        ElementTypesSwitch<Type_t::i16, Type_t::i32, Type_t::i64, Type_t::u8, Type_t::u16, Type_t::u32, Type_t::u64>;
    ElementwiseTypeSwitch switchObj{op_type_, max_threads_per_block_};
    SupportedElementTypes::switch_(element_type_, switchObj, stream, in0, in0_num_elements, in1, in1_num_elements, out);
}

}  // namespace kernel
}  // namespace CUDAPlugin

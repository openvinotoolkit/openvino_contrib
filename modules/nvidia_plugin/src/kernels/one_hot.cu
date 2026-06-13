// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <fmt/format.h>

#include <cuda/float16.hpp>

#include "details/type_validator.hpp"
#include "one_hot.hpp"

namespace ov {
namespace nvidia_gpu {
namespace kernel {

template <typename T, typename Idx>
static __global__ void one_hot_kernel(const Idx* indices,
                                      const T* on_value,
                                      const T* off_value,
                                      T* output,
                                      const std::size_t depth,
                                      const std::size_t inner_size,
                                      const std::size_t total) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    const std::size_t depth_inner = depth * inner_size;
    const std::size_t outer = idx / depth_inner;
    const std::size_t rem = idx % depth_inner;
    const std::size_t position = rem / inner_size;
    const std::size_t inner = rem % inner_size;

    const long long index_value = static_cast<long long>(indices[outer * inner_size + inner]);
    output[idx] = (index_value == static_cast<long long>(position)) ? on_value[0] : off_value[0];
}

OneHot::OneHot(Type_t element_type,
               Type_t indices_type,
               std::size_t outer_size,
               std::size_t depth,
               std::size_t inner_size,
               std::size_t num_blocks,
               std::size_t threads_per_block)
    : element_type_{element_type},
      indices_type_{indices_type},
      outer_size_{outer_size},
      depth_{depth},
      inner_size_{inner_size},
      num_blocks_{num_blocks},
      threads_per_block_{threads_per_block} {
    TypeValidator<AllElementTypesSwitch>::check(element_type_);
    TypeValidator<ElementTypesSwitch<Type_t::i32, Type_t::i64>>::check(indices_type_);
}

void OneHot::operator()(cudaStream_t stream,
                        const void* indices,
                        const void* on_value,
                        const void* off_value,
                        void* output) const {
    switch (indices_type_) {
        case Type_t::i32:
            return callByIndexType<int32_t>(stream, indices, on_value, off_value, output);
        case Type_t::i64:
            return callByIndexType<int64_t>(stream, indices, on_value, off_value, output);
        default:
            throw_ov_exception(fmt::format("Index type = {} is not supported by OneHot operation !!", indices_type_));
    }
}

template <typename Idx>
void OneHot::callByIndexType(cudaStream_t stream,
                             const void* indices,
                             const void* on_value,
                             const void* off_value,
                             void* output) const {
    switch (element_type_) {
        case Type_t::boolean:
            return call<bool, Idx>(stream, indices, on_value, off_value, output);
#ifdef CUDA_HAS_BF16_TYPE
        case Type_t::bf16:
            return call<__nv_bfloat16, Idx>(stream, indices, on_value, off_value, output);
#endif
        case Type_t::f16:
            return call<__half, Idx>(stream, indices, on_value, off_value, output);
        case Type_t::f32:
            return call<float, Idx>(stream, indices, on_value, off_value, output);
        case Type_t::f64:
            return call<double, Idx>(stream, indices, on_value, off_value, output);
        case Type_t::i8:
            return call<int8_t, Idx>(stream, indices, on_value, off_value, output);
        case Type_t::i16:
            return call<int16_t, Idx>(stream, indices, on_value, off_value, output);
        case Type_t::i32:
            return call<int32_t, Idx>(stream, indices, on_value, off_value, output);
        case Type_t::i64:
            return call<int64_t, Idx>(stream, indices, on_value, off_value, output);
        case Type_t::u8:
            return call<uint8_t, Idx>(stream, indices, on_value, off_value, output);
        case Type_t::u16:
            return call<uint16_t, Idx>(stream, indices, on_value, off_value, output);
        case Type_t::u32:
            return call<uint32_t, Idx>(stream, indices, on_value, off_value, output);
        case Type_t::u64:
            return call<uint64_t, Idx>(stream, indices, on_value, off_value, output);
        default:
            throw_ov_exception(fmt::format("Element type = {} is not supported by OneHot operation !!", element_type_));
    }
}

template <typename T, typename Idx>
void OneHot::call(cudaStream_t stream,
                  const void* indices,
                  const void* on_value,
                  const void* off_value,
                  void* output) const {
    const std::size_t total = outer_size_ * depth_ * inner_size_;
    if (total == 0) {
        return;
    }
    one_hot_kernel<T, Idx><<<num_blocks_, threads_per_block_, 0, stream>>>(static_cast<const Idx*>(indices),
                                                                           static_cast<const T*>(on_value),
                                                                           static_cast<const T*>(off_value),
                                                                           static_cast<T*>(output),
                                                                           depth_,
                                                                           inner_size_,
                                                                           total);
    throwIfError(cudaPeekAtLastError());
}

}  // namespace kernel
}  // namespace nvidia_gpu
}  // namespace ov

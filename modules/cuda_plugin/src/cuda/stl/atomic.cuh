// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cuda/runtime.hpp>

namespace CUDA {

#if __CUDA_ARCH__ < 600
namespace details {
template <typename T>
struct Plus {
    __device__ T operator()(const T& first, const T& second) const { return first + second; }
};

template <typename T>
struct Minus {
    __device__ T operator()(const T& first, const T& second) const { return first - second; }
};

template <typename T>
struct BitAnd {
    __device__ T operator()(const T& first, const T& second) const { return first & second; }
};

template <typename T>
struct BitOr {
    __device__ T operator()(const T& first, const T& second) const { return first | second; }
};

template <typename T>
struct BitXor {
    __device__ T operator()(const T& first, const T& second) const { return first ^ second; }
};

template <typename T>
struct Exch {
    __device__ T operator()(const T& first, const T&) const { return first; }
};

template <typename Op>
__device__ int atomicOp(int* address, const int val) {
    auto address_as_ull = address;
    int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, Op{}(val, assumed));
        // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return old;
}

template <typename Op>
__device__ unsigned int atomicOp(unsigned int* address, const unsigned int val) {
    auto address_as_ull = address;
    unsigned int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, Op{}(val, assumed));
        // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return old;
}

template <typename Op>
__device__ unsigned long long atomicOp(unsigned long long* address, const unsigned long long val) {
    auto address_as_ull = address;
    unsigned long long old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, Op{}(val, assumed));
        // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return old;
}

template <typename Op>
__device__ float atomicOp(float* address, const float val) {
    auto address_as_ui = reinterpret_cast<int*>(address);
    int old = *address_as_ui, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ui, assumed, __float_as_uint(Op{}(val, __int_as_float(assumed))));
        // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __int_as_float(old);
}

template <typename Op>
__device__ double atomicOp(double* address, const double val) {
    auto address_as_ull = reinterpret_cast<long long*>(address);
    long long old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(Op{}(val, __longlong_as_double(assumed))));
        // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}

}  // namespace details

template <typename T>
__device__ T atomicAdd_system(T* address, T val) {
    return details::atomicOp<details::Plus<T>>(address, val);
}

template <typename T>
__device__ T atomicSub_system(T* address, T val) {
    return details::atomicOp<details::Minus<T>>(address, val);
}

template <typename T>
__device__ T atomicAnd_system(T* address, T val) {
    return details::atomicOp<details::BitAnd<T>>(address, val);
}

template <typename T>
__device__ T atomicOr_system(T* address, T val) {
    return details::atomicOp<details::BitOr<T>>(address, val);
}

template <typename T>
__device__ T atomicXor_system(T* address, T val) {
    return details::atomicOp<details::BitXor<T>>(address, val);
}

template <typename T>
__device__ T atomicExch_system(T* address, const T val) {
    return details::atomicOp<details::Exch<T>>(address, val);
}
#endif

template <typename T>
__device__ T atomicLoad(const T* addr) {
    // To bypass cache
    const volatile T* vaddr = addr;
    // For seq_cst loads. Remove for acquire semantics.
    __threadfence();
    const T value = *vaddr;
    // Fence to ensure that dependent reads are correctly ordered
    __threadfence();
    return value;
}

template <typename T>
__device__ T atomicLoad_block(const T* addr) {
    // To bypass cache
    const volatile T* vaddr = addr;
    // For seq_cst loads. Remove for acquire semantics.
    __threadfence_block();
    const T value = *vaddr;
    // Fence to ensure that dependent reads are correctly ordered
    __threadfence_block();
    return value;
}

template <typename T>
__device__ T atomicLoad_system(const T* addr) {
    // To bypass cachesssss
    const volatile T* vaddr = addr;
    // For seq_cst loads. Remove for acquire semantics.
    __threadfence_system();
    const T value = *vaddr;
    // Fence to ensure that dependent reads are correctly ordered
    __threadfence_system();
    return value;
}

template <typename T>
__device__ void atomicStore(T* addr, const T value) {
    // To bypass cache
    volatile T* vaddr = addr;
    // Fence to ensure that previous non-atomic stores are visible to other threads
    __threadfence();
    *vaddr = value;
}

template <typename T>
__device__ void atomicStore_block(T* addr, const T value) {
    // To bypass cache
    volatile T* vaddr = addr;
    // Fence to ensure that previous non-atomic stores are visible to other threads
    __threadfence_block();
    *vaddr = value;
}

template <typename T>
__device__ void atomicStore_system(T* addr, const T value) {
    volatile T* vaddr = addr;  // To bypass cache
    // Fence to ensure that previous non-atomic stores are visible to other threads
    __threadfence_system();
    *vaddr = value;
}

enum class AtomicScope { DeviceWide, BlockWide, SystemWide };

template <AtomicScope Scope = AtomicScope::DeviceWide>
struct AtomicOperations;

template <>
struct AtomicOperations<AtomicScope::DeviceWide> {
    template <typename T>
    __device__ static T load(const T& value) noexcept {
        return atomicLoad(&value);
    }
    template <typename T>
    __device__ static void store(T& value, const T& val) noexcept {
        atomicStore(&value, val);
    }
    template <typename T>
    __device__ static T add(T& value, const T& val) noexcept {
        return atomicAdd(&value, val);
    }
    template <typename T>
    __device__ static T sub(T& value, const T& val) noexcept {
        return atomicSub(&value, val);
    }
    template <typename T>
    __device__ static T bitAnd(T& value, const T& val) noexcept {
        return atomicAnd(&value, val);
    }
    template <typename T>
    __device__ static T bitOr(T& value, const T& val) noexcept {
        return atomicOr(&value, val);
    }
    template <typename T>
    __device__ static T bitXor(T& value, const T& val) noexcept {
        return atomicXor(&value, val);
    }
    template <typename T>
    __device__ static T exch(T& value, const T& val) noexcept {
        return atomicExch(&value, val);
    }
    template <typename T>
    __device__ static T cas(T& value, const T old, const T& val) noexcept {
        return atomicCAS(&value, old, val);
    }
};

template <>
struct AtomicOperations<AtomicScope::BlockWide> {
    template <typename T>
    __device__ static T load(const T& value) noexcept {
        return CUDA::atomicLoad_block(&value);
    }
    template <typename T>
    __device__ static void store(T& value, const T& val) noexcept {
        atomicStore_block(&value, val);
    }
    template <typename T>
    __device__ static T add(T& value, const T& val) noexcept {
        return atomicAdd_block(&value, val);
    }
    template <typename T>
    __device__ static T sub(T& value, const T& val) noexcept {
        return atomicSub_block(&value, val);
    }
    template <typename T>
    __device__ static T bitAnd(T& value, const T& val) noexcept {
        return atomicAnd_block(&value, val);
    }
    template <typename T>
    __device__ static T bitOr(T& value, const T& val) noexcept {
        return atomicOr_block(&value, val);
    }
    template <typename T>
    __device__ static T bitXor(T& value, const T& val) noexcept {
        return atomicXor_block(&value, val);
    }
    template <typename T>
    __device__ static T exch(T& value, const T& val) noexcept {
        return atomicExch_block(&value, val);
    }
    template <typename T>
    __device__ static T cas(T& value, const T old, const T& val) noexcept {
        return atomicCAS_block(&value, old, val);
    }
};

template <>
struct AtomicOperations<AtomicScope::SystemWide> {
    template <typename T>
    __device__ static T load(const T& value) noexcept {
        return CUDA::atomicLoad_system(&value);
    }
    template <typename T>
    __device__ static void store(T& value, const T& val) noexcept {
        atomicStore_system(&value, val);
    }
    template <typename T>
    __device__ static T add(T& value, const T& val) noexcept {
        return atomicAdd_system(&value, val);
    }
    template <typename T>
    __device__ static T sub(T& value, const T& val) noexcept {
        return atomicSub_system(&value, val);
    }
    template <typename T>
    __device__ static T bitAnd(T& value, const T& val) noexcept {
        return atomicAnd_system(&value, val);
    }
    template <typename T>
    __device__ static T bitOr(T& value, const T& val) noexcept {
        return atomicOr_system(&value, val);
    }
    template <typename T>
    __device__ static T bitXor(T& value, const T& val) noexcept {
        return atomicXor_system(&value, val);
    }
    template <typename T>
    __device__ static T exch(T& value, const T& val) noexcept {
        return atomicExch_system(&value, val);
    }
    template <typename T>
    __device__ static T cas(T& value, const T old, const T& val) noexcept {
        return atomicCAS_system(&value, old, val);
    }
};

/**
 * Atomic implementation for CUDA.
 * Implementation is compatible with standard C++ std::atomic,
 * except no memory_order is available
 * @tparam T Type for atomic
 * @tparam Scope Scope for CUDA atomic, see @ref AtomicScope
 */
template <typename T, AtomicScope Scope = AtomicScope::DeviceWide>
class Atomic {
public:
    __device__ Atomic() noexcept = default;
    __device__ constexpr Atomic(T desired) noexcept : value_{desired} {}

    __device__ T operator=(const T desired) noexcept {
        AtomicOperations<Scope>::exch(value_, desired);
        return desired;
    }

    __device__ T operator=(const T desired) volatile noexcept {
        AtomicOperations<Scope>::exch(value_, desired);
        return desired;
    }

    __device__ constexpr Atomic(const Atomic& other) = delete;
    __device__ Atomic& operator=(const Atomic& other) = delete;
    __device__ Atomic& operator=(const Atomic& other) volatile = delete;

    __device__ T load() const noexcept { return AtomicOperations<Scope>::load(value_); }

    __device__ void store(const T val) noexcept { AtomicOperations<Scope>::exch(value_, val); }

    __device__ T exchange(const T desired) noexcept { return AtomicOperations<Scope>::exch(value_, desired); }

    __device__ bool compare_exchange_weak(T& expected, const T desired) noexcept {
        const T old = AtomicOperations<Scope>::cas(value_, expected, desired);
        const bool success = old == expected;
        expected = old;
        return success;
    }

    __device__ T fetch_add(const T val) noexcept { return AtomicOperations<Scope>::add(value_, val); }

    __device__ T fetch_sub(const T val) noexcept { return AtomicOperations<Scope>::sub(value_, val); }

    __device__ T fetch_and(const T val) noexcept { return AtomicOperations<Scope>::bitAnd(value_, val); }

    __device__ T fetch_or(const T val) noexcept { return AtomicOperations<Scope>::bitOr(value_, val); }

    __device__ T fetch_xor(const T val) noexcept { return AtomicOperations<Scope>::bitXor(value_, val); }

    __device__ T operator++() noexcept { return fetch_add(1) + 1; }

    __device__ T operator++(int) noexcept { return fetch_add(1); }

    __device__ T operator--() noexcept { return fetch_sub(1); }

    __device__ T operator--(int) noexcept { return fetch_sub(1) - 1; }

    __device__ T operator+=(const T val) noexcept { return fetch_add(val) + val; }

    __device__ T operator-=(const T val) noexcept { return fetch_sub(val) + val; }

    __device__ T operator&=(const T val) noexcept { return fetch_and(val) + val; }

    __device__ T operator|=(const T val) noexcept { return fetch_or(val) + val; }

    __device__ T operator^=(const T val) noexcept { return fetch_xor(val) + val; }

private:
    alignas(memoryAlignment) T value_{};
};

template <typename T>
using DeviceAtomic = Atomic<T, CUDA::AtomicScope::DeviceWide>;

template <typename T>
using BlockAtomic = Atomic<T, CUDA::AtomicScope::BlockWide>;

template <typename T>
using SystemAtomic = Atomic<T, CUDA::AtomicScope::SystemWide>;

}  // namespace CUDA

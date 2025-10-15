// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cuda/runtime.hpp>
#include <cuda/stl/mdvector.cuh>

using namespace ov::nvidia_gpu;

class MDVectorTest : public testing::Test {
    void SetUp() override {}

    void TearDown() override {}
};

namespace {

template <size_t VectorCapacity, typename T>
__global__ void push_back(CUDA::MDVector<T, 2> mdvec) {
    auto vec = mdvec(0, 1);
    assert(vec.capacity() == VectorCapacity);
    assert(vec.size() == 0);
    for (int i = 0; i < VectorCapacity; ++i) {
        vec.push_back(i);
    }
}

template <size_t VectorCapacity, typename T>
__global__ void verify_push_back(CUDA::MDVector<T, 2> mdvec) {
    auto vec0 = mdvec(0, 0);
    auto vec1 = mdvec(0, 1);
    assert(vec0.size() == 0);
    assert(vec1.size() == VectorCapacity);
}

template <size_t VectorCapacity, typename T>
__global__ void erase(CUDA::MDVector<T, 2> mdvec) {
    auto vec = mdvec(0, 1);
    T* erase_element = nullptr;
    for (int fe = 0; fe < VectorCapacity; ++fe) {
        if (fe == 3) {
            erase_element = vec.data() + fe;
            break;
        }
    }
    vec.erase(erase_element);
    erase_element = nullptr;
    for (int fe = 0; fe < VectorCapacity - 1; ++fe) {
        if (fe == 14) {
            erase_element = vec.data() + fe;
            break;
        }
    }
    vec.erase(erase_element);
}

template <size_t VectorCapacity, typename T>
__global__ void verify_erase(CUDA::MDVector<T, 2> vec) {
    assert(vec(0, 0).size() == 0);
    assert(vec(0, 1).size() == VectorCapacity - 2);
}

}  // namespace

TEST_F(MDVectorTest, MDVector_PushBack) {
    using VectorTestType = CUDA::MDVector<int, 2>;

    constexpr auto kVectorCapacity = 1023ul;

    CUDA::Stream stream{};
    auto src = stream.malloc(VectorTestType::size_of(kVectorCapacity, 2, 8));
    stream.memset(src, 0, VectorTestType::size_of(kVectorCapacity, 2, 8));
    auto vec = VectorTestType(stream, kVectorCapacity, src.get(), 2, 8);
    push_back<kVectorCapacity><<<1, 1, 0, stream.get()>>>(vec);
    verify_push_back<kVectorCapacity><<<1, 1, 0, stream.get()>>>(vec);
    ASSERT_NO_THROW(stream.synchronize());
}

TEST_F(MDVectorTest, MDVector_Erase) {
    using VectorTestType = CUDA::MDVector<int, 2>;

    constexpr auto kVectorCapacity = 872ul;

    CUDA::Stream stream{};
    auto src = stream.malloc(VectorTestType::size_of(kVectorCapacity, 2, 8));
    stream.memset(src, 0, VectorTestType::size_of(kVectorCapacity, 2, 8));
    auto vec = VectorTestType(stream, kVectorCapacity, src.get(), 2, 8);
    push_back<kVectorCapacity><<<1, 1, 0, stream.get()>>>(vec);
    verify_push_back<kVectorCapacity><<<1, 1, 0, stream.get()>>>(vec);
    erase<kVectorCapacity><<<1, 1, 0, stream.get()>>>(vec);
    verify_erase<kVectorCapacity><<<1, 1, 0, stream.get()>>>(vec);
    ASSERT_NO_THROW(stream.synchronize());
}

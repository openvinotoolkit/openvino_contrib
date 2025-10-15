// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cuda/runtime.hpp>
#include <cuda/stl/vector.cuh>

using namespace ov::nvidia_gpu;

class VectorTest : public testing::Test {
    void SetUp() override {}

    void TearDown() override {}
};

namespace {

template <size_t VectorCapacity, typename T>
__global__ void push_back(CUDA::Vector<T> vec) {
    assert(vec.capacity() == VectorCapacity);
    assert(vec.size() == 0);
    for (int i = 0; i < VectorCapacity; ++i) {
        vec.push_back(i);
    }
}

template <size_t VectorCapacity, typename T>
__global__ void verify_push_back(CUDA::Vector<T> vec) {
    assert(vec.size() == VectorCapacity);
}

template <size_t VectorCapacity, typename T>
__global__ void erase(CUDA::Vector<T> vec) {
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
__global__ void verify_erase(CUDA::Vector<T> vec) {
    assert(vec.size() == VectorCapacity - 2);
}

}  // namespace

TEST_F(VectorTest, Vector_PushBack) {
    using VectorTestType = CUDA::Vector<int>;

    constexpr auto kVectorCapacity = 1023ul;

    CUDA::Stream stream{};
    auto src = stream.malloc(VectorTestType::size_of(kVectorCapacity));
    stream.memset(src, 0, VectorTestType::size_of(kVectorCapacity));
    auto vec = CUDA::Vector<int>(src.get(), kVectorCapacity);
    push_back<kVectorCapacity><<<1, 1, 0, stream.get()>>>(vec);
    verify_push_back<kVectorCapacity><<<1, 1, 0, stream.get()>>>(vec);
    ASSERT_NO_THROW(stream.synchronize());
}

TEST_F(VectorTest, Vector_Erase) {
    using VectorTestType = CUDA::Vector<int>;

    constexpr auto kVectorCapacity = 524ul;

    CUDA::Stream stream{};
    auto src = stream.malloc(VectorTestType::size_of(kVectorCapacity));
    stream.memset(src, 0, VectorTestType::size_of(kVectorCapacity));
    auto vec = VectorTestType(src.get(), kVectorCapacity);
    push_back<kVectorCapacity><<<1, 1, 0, stream.get()>>>(vec);
    verify_push_back<kVectorCapacity><<<1, 1, 0, stream.get()>>>(vec);
    erase<kVectorCapacity><<<1, 1, 0, stream.get()>>>(vec);
    verify_erase<kVectorCapacity><<<1, 1, 0, stream.get()>>>(vec);
    ASSERT_NO_THROW(stream.synchronize());
}

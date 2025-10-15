// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cuda/runtime.hpp>
#include <cuda/stl/atomic.cuh>

using namespace ov::nvidia_gpu;

class AtomicTest : public testing::Test {
    void SetUp() override {}

    void TearDown() override {}
};

namespace {

template <typename T>
__global__ void init(CUDA::DeviceAtomic<T>& count, T value) {
    count = value;
}

template <typename T>
__global__ void add(CUDA::DeviceAtomic<T>& count) {
    count += 1;
}

template <typename T>
__global__ void sub(CUDA::DeviceAtomic<T>& count) {
    count -= 1;
}

template <typename T>
__global__ void cas(CUDA::DeviceAtomic<T>& count) {
    const auto id = threadIdx.x;
    unsigned prevId = 0;
    while (true) {
        if (count.compare_exchange_weak(prevId, id)) {
            break;
        }
    }
}

}  // namespace

TEST_F(AtomicTest, Atomic_Add) {
    const auto atomicSize = sizeof(CUDA::DeviceAtomic<unsigned>);

    const auto& defaultStream = CUDA::DefaultStream::stream();
    auto src = defaultStream.malloc(atomicSize);

    constexpr auto kGridNum = 1024;
    constexpr auto kBlockNum = 1024;

    CUDA::Stream stream{};
    auto& accumAtomic = (*static_cast<CUDA::DeviceAtomic<unsigned>*>(src.get()));
    init<<<1, 1, 0, stream.get()>>>(accumAtomic, 0u);
    add<<<kGridNum, kBlockNum, 0, stream.get()>>>(accumAtomic);
    stream.synchronize();

    unsigned countHost = 0;
    defaultStream.download(&countHost, CUDA::DevicePointer<const void*>{src.get()}, sizeof(unsigned));
    ASSERT_EQ(countHost, kGridNum * kBlockNum);
}

TEST_F(AtomicTest, Atomic_AddBias) {
    const auto atomicSize = sizeof(CUDA::DeviceAtomic<unsigned>);

    const auto& defaultStream = CUDA::DefaultStream::stream();
    auto src = defaultStream.malloc(atomicSize);

    constexpr auto kGridNum = 1024;
    constexpr auto kBlockNum = 1024;

    CUDA::Stream stream{};
    auto& accumAtomic = (*static_cast<CUDA::DeviceAtomic<unsigned>*>(src.get()));
    constexpr auto biasAtomicValue = 11u;
    init<<<1, 1, 0, stream.get()>>>(accumAtomic, biasAtomicValue);
    add<<<kGridNum, kBlockNum, 0, stream.get()>>>(accumAtomic);
    stream.synchronize();

    unsigned countHost = 0;
    defaultStream.download(&countHost, CUDA::DevicePointer<const void*>{src.get()}, sizeof(unsigned));
    ASSERT_EQ(countHost, biasAtomicValue + kGridNum * kBlockNum);
}

TEST_F(AtomicTest, Atomic_Sub) {
    const auto atomicSize = sizeof(CUDA::DeviceAtomic<int>);

    const auto& defaultStream = CUDA::DefaultStream::stream();
    auto src = defaultStream.malloc(atomicSize);

    constexpr auto kGridNum = 1024;
    constexpr auto kBlockNum = 1024;

    CUDA::Stream stream{};
    auto& accumAtomic = (*static_cast<CUDA::DeviceAtomic<int>*>(src.get()));
    init<<<1, 1, 0, stream.get()>>>(accumAtomic, 0);
    sub<<<kGridNum, kBlockNum, 0, stream.get()>>>(accumAtomic);
    stream.synchronize();

    unsigned countHost = 0;
    defaultStream.download(&countHost, CUDA::DevicePointer<const void*>{src.get()}, sizeof(int));
    ASSERT_EQ(countHost, -(kGridNum * kBlockNum));
}

TEST_F(AtomicTest, Atomic_SubBias) {
    const auto atomicSize = sizeof(CUDA::DeviceAtomic<int>);

    const auto& defaultStream = CUDA::DefaultStream::stream();
    auto src = defaultStream.malloc(atomicSize);

    constexpr auto kGridNum = 1024;
    constexpr auto kBlockNum = 1024;

    CUDA::Stream stream{};
    auto& accumAtomic = (*static_cast<CUDA::DeviceAtomic<int>*>(src.get()));
    constexpr auto biasAtomicValue = 1041254;
    init<<<1, 1, 0, stream.get()>>>(accumAtomic, biasAtomicValue);
    sub<<<kGridNum, kBlockNum, 0, stream.get()>>>(accumAtomic);
    stream.synchronize();

    unsigned countHost = 0;
    defaultStream.download(&countHost, CUDA::DevicePointer<const void*>{src.get()}, sizeof(int));
    ASSERT_EQ(countHost, biasAtomicValue - (kGridNum * kBlockNum));
}

TEST_F(AtomicTest, Atomic_CAS) {
    const auto atomicSize = sizeof(CUDA::DeviceAtomic<unsigned>);

    const auto& defaultStream = CUDA::DefaultStream::stream();
    auto src = defaultStream.malloc(atomicSize);

    constexpr auto kGridNum = 1024;
    constexpr auto kBlockNum = 1024;

    CUDA::Stream stream{};
    auto& accumAtomic = (*static_cast<CUDA::DeviceAtomic<unsigned>*>(src.get()));
    init<<<1, 1, 0, stream.get()>>>(accumAtomic, 0u);
    cas<<<kGridNum, kBlockNum, 0, stream.get()>>>(accumAtomic);
    stream.synchronize();

    unsigned valueHost = 0;
    defaultStream.download(&valueHost, CUDA::DevicePointer<const void*>{src.get()}, sizeof(int));
    ASSERT_LT(valueHost, kGridNum * kBlockNum);
}

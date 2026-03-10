// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <chrono>
#include <cuda/runtime.hpp>
#include <cuda/stl/algorithms/sort.cuh>
#include <random>

using namespace ov::nvidia_gpu;

namespace {

template <typename T, typename Compare>
__global__ void quick_sort_launch(T* first, T* last, int* stack, Compare comparer) {
    CUDA::algorithms::quick_sort_iterative(first, last, stack, comparer);
}

template <typename T, typename Compare>
__global__ void quick_sort_launch(T* first, T* last, Compare comparer) {
    CUDA::algorithms::quick_sort_iterative(first, last, comparer);
}

template <typename T, typename Compare>
__global__ void partial_quick_sort_launch(T* first, T* last, int* stack, const size_t topk, Compare comparer) {
    CUDA::algorithms::partial_quick_sort_iterative(first, last, stack, topk, comparer);
}

template <typename T, typename Compare>
__global__ void partial_quick_sort_launch(T* first, T* last, const size_t topk, Compare comparer) {
    CUDA::algorithms::partial_quick_sort_iterative(first, last, topk, comparer);
}

template <typename T, typename Compare>
__global__ void parallel_quick_sort_launch(T* first, T* last, int* stack, Compare comparer) {
    const auto size = last - first;
    const auto num_chunks = (size % gridDim.x == 0) ? gridDim.x : (gridDim.x + 1);
    const auto chunk_size = size / num_chunks;
    auto begin = first + blockIdx.x * chunk_size;
    auto end = first + (blockIdx.x + 1) * chunk_size;
    if (end > last) {
        end = last;
    }
    auto stack_begin = stack + blockIdx.x * chunk_size;
    constexpr auto kNumPartitions = 1024;
    CUDA::algorithms::parallel::quick_sort_iterative<kNumPartitions>(
        begin, end, stack_begin, comparer, threadIdx.x, blockDim.x);
}

template <typename T, typename Compare>
__global__ void parallel_quick_sort_launch(T* first, T* last, Compare comparer) {
    const auto size = last - first;
    const auto num_chunks = (size % gridDim.x == 0) ? gridDim.x : (gridDim.x + 1);
    const auto chunk_size = size / num_chunks;
    auto begin = first + blockIdx.x * chunk_size;
    auto end = first + (blockIdx.x + 1) * chunk_size;
    if (end > last) {
        end = last;
    }
    constexpr auto kNumPartitions = 1024;
    CUDA::algorithms::parallel::quick_sort_iterative<kNumPartitions>(begin, end, comparer, threadIdx.x, blockDim.x);
}

}  // namespace

class QuickSortKernelTest : public testing::Test {
    void SetUp() override {}

    void TearDown() override {}

public:
    template <size_t ArrayLength, bool WithStack>
    void runQuickSort() {
        auto arr = std::make_unique<float[]>(ArrayLength);
        auto sortedArr = std::make_unique<float[]>(ArrayLength);
        std::random_device rd{};
        std::mt19937 gen{rd()};
        std::uniform_real_distribution<float> dist(-20000000, 20000000);

        constexpr size_t arr_size = sizeof(float) * ArrayLength;

        for (int n = 0; n < ArrayLength; ++n) {
            arr[n] = dist(gen);
        }

        const auto& defaultStream = CUDA::DefaultStream::stream();
        auto src = defaultStream.malloc(arr_size);
        defaultStream.upload(src, arr.get(), arr_size);
        auto begin = static_cast<float*>(src.get());
        auto end = begin + ArrayLength;
        CUDA::Stream stream{};
        if (WithStack) {
            auto stack = defaultStream.malloc(arr_size);
            auto stack_begin = static_cast<int*>(stack.get());
            quick_sort_launch<<<1, 1, 0, stream.get()>>>(begin, end, stack_begin, CUDA::algorithms::Less<float>{});
        } else {
            quick_sort_launch<<<1, 1, 0, stream.get()>>>(begin, end, CUDA::algorithms::Less<float>{});
        }
        stream.synchronize();
        defaultStream.download(sortedArr.get(), src, arr_size);

        std::sort(arr.get(), arr.get() + ArrayLength);
        for (int i = 0; i < ArrayLength; ++i) {
            if (sortedArr[i] != arr[i]) {
                std::cout << "i = " << i << std::endl;
                std::cout << "arr[i] = " << arr[i] << std::endl;
                std::cout << "sortedArr[i] = " << sortedArr[i] << std::endl;
            }
            ASSERT_FLOAT_EQ(sortedArr[i], arr[i]);
        }
    }

    template <size_t ArrayLength, size_t TopK, bool WithStack>
    void runPartialQuickSort() {
        auto arr = std::make_unique<float[]>(ArrayLength);
        auto sortedArr = std::make_unique<float[]>(ArrayLength);
        std::random_device rd{};
        std::mt19937 gen{rd()};
        std::uniform_real_distribution<float> dist(-20000000, 20000000);

        constexpr size_t arr_size = sizeof(float) * ArrayLength;

        for (int n = 0; n < ArrayLength; ++n) {
            arr[n] = dist(gen);
        }

        const auto& defaultStream = CUDA::DefaultStream::stream();
        auto src = defaultStream.malloc(arr_size);
        defaultStream.upload(src, arr.get(), arr_size);
        auto begin = static_cast<float*>(src.get());
        auto end = begin + ArrayLength;
        CUDA::Stream stream{};
        if (WithStack) {
            auto stack = defaultStream.malloc(arr_size);
            auto stack_begin = static_cast<int*>(stack.get());
            partial_quick_sort_launch<<<1, 1, 0, stream.get()>>>(
                begin, end, stack_begin, TopK, CUDA::algorithms::Less<float>{});
        } else {
            partial_quick_sort_launch<<<1, 1, 0, stream.get()>>>(begin, end, TopK, CUDA::algorithms::Less<float>{});
        }
        stream.synchronize();
        defaultStream.download(sortedArr.get(), src, arr_size);

        std::sort(arr.get(), arr.get() + ArrayLength);
        for (int i = 0; i < TopK; ++i) {
            if (sortedArr[i] != arr[i]) {
                std::cout << "i = " << i << std::endl;
                std::cout << "arr[i] = " << arr[i] << std::endl;
                std::cout << "sortedArr[i] = " << sortedArr[i] << std::endl;
            }
            ASSERT_FLOAT_EQ(sortedArr[i], arr[i]);
        }
    }

    template <size_t ArrayLength, bool WithStack>
    void runParallelQuickSort() {
        auto arr = std::make_unique<float[]>(ArrayLength);
        auto sortedArr = std::make_unique<float[]>(ArrayLength);
        std::random_device rd{};
        std::mt19937 gen{rd()};
        std::uniform_real_distribution<float> dist(-20000000, 20000000);

        constexpr size_t arr_size = sizeof(float) * ArrayLength;

        for (int n = 0; n < ArrayLength; ++n) {
            arr[n] = dist(gen);
        }

        const auto& defaultStream = CUDA::DefaultStream::stream();
        auto src = defaultStream.malloc(arr_size);
        defaultStream.upload(src, arr.get(), arr_size);
        auto begin = static_cast<float*>(src.get());
        auto end = begin + ArrayLength;
        CUDA::Stream stream{};
        if (WithStack) {
            auto stack = defaultStream.malloc(arr_size);
            auto stack_begin = static_cast<int*>(stack.get());
            parallel_quick_sort_launch<<<1, 100, 0, stream.get()>>>(
                begin, end, stack_begin, CUDA::algorithms::Less<float>{});
        } else {
            parallel_quick_sort_launch<<<1, 100, 0, stream.get()>>>(begin, end, CUDA::algorithms::Less<float>{});
        }
        stream.synchronize();
        defaultStream.download(sortedArr.get(), src, arr_size);

        std::sort(arr.get(), arr.get() + ArrayLength);
        for (int i = 0; i < ArrayLength; ++i) {
            if (sortedArr[i] != arr[i]) {
                std::cout << "i = " << i << std::endl;
                std::cout << "arr[i] = " << arr[i] << std::endl;
                std::cout << "sortedArr[i] = " << sortedArr[i] << std::endl;
            }
            ASSERT_FLOAT_EQ(sortedArr[i], arr[i]);
        }
    }
};

TEST_F(QuickSortKernelTest, QuickSortKernel_0) { runQuickSort<650023, true>(); }

TEST_F(QuickSortKernelTest, QuickSortKernelWithoutStack) { runQuickSort<650023, false>(); }

TEST_F(QuickSortKernelTest, PartialQuickSortKernel_0) { runPartialQuickSort<650023, 100, true>(); }

TEST_F(QuickSortKernelTest, PartialQuickSortKernelWithoutStack_0) { runPartialQuickSort<650023, 100, false>(); }

TEST_F(QuickSortKernelTest, PartialQuickSortKernel_1) { runPartialQuickSort<4214235, 100, true>(); }

TEST_F(QuickSortKernelTest, PartialQuickSortKernelWithoutStack_1) { runPartialQuickSort<4214235, 100, false>(); }

TEST_F(QuickSortKernelTest, PartialQuickSortKernel_2) { runPartialQuickSort<650023, 251, true>(); }

TEST_F(QuickSortKernelTest, PartialQuickSortKernelWithoutStack_2) { runPartialQuickSort<650023, 251, false>(); }

TEST_F(QuickSortKernelTest, PartialQuickSortKernel_3) { runPartialQuickSort<4214235, 521, true>(); }

TEST_F(QuickSortKernelTest, PartialQuickSortKernelWithoutStack_3) { runPartialQuickSort<4214235, 521, false>(); }

TEST_F(QuickSortKernelTest, ParallelQuickSortKernel_0) { runParallelQuickSort<650023, true>(); }

TEST_F(QuickSortKernelTest, ParallelQuickSortKernelWithoutStack_0) { runParallelQuickSort<650023, false>(); }

TEST_F(QuickSortKernelTest, ParallelQuickSortKernel_1) { runParallelQuickSort<4214235, true>(); }

TEST_F(QuickSortKernelTest, ParallelQuickSortKernelWithoutStack_1) { runParallelQuickSort<4214235, false>(); }

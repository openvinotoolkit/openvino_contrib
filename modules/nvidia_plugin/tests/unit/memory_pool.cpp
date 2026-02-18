// Copyright (C) 2020-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <thread>

#include "memory_manager/cuda_memory_pool.hpp"
#include "memory_manager/model/cuda_memory_model.hpp"

using namespace ov::nvidia_gpu;

class MemoryPoolTest : public testing::Test {
    void SetUp() override {}
    void TearDown() override {}

public:
    size_t GetNumAvailableMemoryManagers(MemoryPool& pool) { return pool.memory_blocks_.size(); }

    MemoryPool::DynamicHandle CallAllocateDynamic(MemoryPool& pool, size_t bytes, CancellationToken& token) {
        return pool.AllocateDynamic(bytes, token);
    }

    void CallReleaseDynamicChunk(MemoryPool& pool, DynamicChunk chunk) {
        pool.ReleaseDynamicChunk(std::move(chunk));
    }

    size_t GetPendingRequestCount(MemoryPool& pool) {
        return pool.pending_requests_.size();
    }

    void AddPendingRequest(MemoryPool& pool, size_t size, uint64_t id) {
        std::lock_guard<std::mutex> lock{pool.dyn_mtx_};
        pool.pending_requests_.push_back({size, id, false, std::nullopt});
    }

    bool IsPendingRequestDone(MemoryPool& pool, size_t index) {
        std::lock_guard<std::mutex> lock{pool.dyn_mtx_};
        return pool.pending_requests_[index].done;
    }

    bool PendingRequestHasChunk(MemoryPool& pool, size_t index) {
        std::lock_guard<std::mutex> lock{pool.dyn_mtx_};
        return pool.pending_requests_[index].chunk.has_value();
    }

    size_t GetPendingChunkUsableSize(MemoryPool& pool, size_t index) {
        std::lock_guard<std::mutex> lock{pool.dyn_mtx_};
        return pool.pending_requests_[index].chunk->usable_size;
    }

    size_t GetPendingChunkOffset(MemoryPool& pool, size_t index) {
        std::lock_guard<std::mutex> lock{pool.dyn_mtx_};
        return pool.pending_requests_[index].chunk->offset;
    }

    void* GetPendingChunkBasePtr(MemoryPool& pool, size_t index) {
        std::lock_guard<std::mutex> lock{pool.dyn_mtx_};
        return pool.pending_requests_[index].chunk->allocation.get();
    }
};

TEST_F(MemoryPoolTest, MemoryManagerProxy_Success) {
    CancellationToken cancellationToken{};
    std::unordered_map<BufferID, ptrdiff_t> offsets;
    auto memoryModel = std::make_shared<MemoryModel>(1000, offsets);
    auto memoryPool = std::make_shared<MemoryPool>(2, memoryModel);
    ASSERT_EQ(GetNumAvailableMemoryManagers(*memoryPool), 2);
    {
        ASSERT_EQ(GetNumAvailableMemoryManagers(*memoryPool), 2);
        auto memoryManagerProxy = memoryPool->WaitAndGet(cancellationToken);
        ASSERT_EQ(GetNumAvailableMemoryManagers(*memoryPool), 1);
    }
    ASSERT_EQ(GetNumAvailableMemoryManagers(*memoryPool), 2);

    {
        ASSERT_EQ(GetNumAvailableMemoryManagers(*memoryPool), 2);
        auto memoryManagerProxy0 = memoryPool->WaitAndGet(cancellationToken);
        ASSERT_EQ(GetNumAvailableMemoryManagers(*memoryPool), 1);
        auto memoryManagerProxy1 = memoryPool->WaitAndGet(cancellationToken);
        ASSERT_EQ(GetNumAvailableMemoryManagers(*memoryPool), 0);
    }
    ASSERT_EQ(GetNumAvailableMemoryManagers(*memoryPool), 2);
}

// --- Dynamic allocation tests ---

TEST_F(MemoryPoolTest, DynamicAlloc_BasicAllocAndFree) {
    CancellationToken token{};
    std::unordered_map<BufferID, ptrdiff_t> offsets;
    auto memoryModel = std::make_shared<MemoryModel>(1000, offsets);
    auto pool = std::make_shared<MemoryPool>(1, memoryModel);
    auto proxy = pool->WaitAndGet(token);

    {
        auto handle = proxy.AllocateDynamic(512, token);
        ASSERT_TRUE(static_cast<bool>(handle));
        ASSERT_NE(handle.get(), nullptr);
        ASSERT_GE(handle.size(), 512u);
    }
    // Handle destroyed — memory returned to pool
}

TEST_F(MemoryPoolTest, DynamicAlloc_MultipleAllocations) {
    CancellationToken token{};
    std::unordered_map<BufferID, ptrdiff_t> offsets;
    auto memoryModel = std::make_shared<MemoryModel>(1000, offsets);
    auto pool = std::make_shared<MemoryPool>(1, memoryModel);
    auto proxy = pool->WaitAndGet(token);

    auto handle1 = proxy.AllocateDynamic(256, token);
    auto handle2 = proxy.AllocateDynamic(512, token);
    auto handle3 = proxy.AllocateDynamic(1024, token);

    ASSERT_NE(handle1.get(), nullptr);
    ASSERT_NE(handle2.get(), nullptr);
    ASSERT_NE(handle3.get(), nullptr);

    // All pointers must be distinct
    ASSERT_NE(handle1.get(), handle2.get());
    ASSERT_NE(handle1.get(), handle3.get());
    ASSERT_NE(handle2.get(), handle3.get());
}

TEST_F(MemoryPoolTest, DynamicAlloc_ZeroSizeThrows) {
    CancellationToken token{};
    std::unordered_map<BufferID, ptrdiff_t> offsets;
    auto memoryModel = std::make_shared<MemoryModel>(1000, offsets);
    auto pool = std::make_shared<MemoryPool>(1, memoryModel);

    ASSERT_THROW(CallAllocateDynamic(*pool, 0, token), ov::Exception);
}

TEST_F(MemoryPoolTest, DynamicAlloc_HandleMoveSemantics) {
    CancellationToken token{};
    std::unordered_map<BufferID, ptrdiff_t> offsets;
    auto memoryModel = std::make_shared<MemoryModel>(1000, offsets);
    auto pool = std::make_shared<MemoryPool>(1, memoryModel);
    auto proxy = pool->WaitAndGet(token);

    auto handle1 = proxy.AllocateDynamic(256, token);
    void* ptr = handle1.get();
    size_t sz = handle1.size();

    auto handle2 = std::move(handle1);
    ASSERT_TRUE(static_cast<bool>(handle2));
    ASSERT_EQ(handle2.get(), ptr);
    ASSERT_EQ(handle2.size(), sz);
}

TEST_F(MemoryPoolTest, DynamicAlloc_SizeIsAligned) {
    CancellationToken token{};
    std::unordered_map<BufferID, ptrdiff_t> offsets;
    auto memoryModel = std::make_shared<MemoryModel>(1000, offsets);
    auto pool = std::make_shared<MemoryPool>(1, memoryModel);

    auto handle = CallAllocateDynamic(*pool, 100, token);
    ASSERT_GE(handle.size(), 256u);
    ASSERT_EQ(handle.size() % CUDA::memoryAlignment, 0u);
}

TEST_F(MemoryPoolTest, DynamicAlloc_InterruptThrows) {
    std::unordered_map<BufferID, ptrdiff_t> offsets;
    auto memoryModel = std::make_shared<MemoryModel>(1000, offsets);
    auto pool = std::make_shared<MemoryPool>(1, memoryModel);

    pool->Interrupt();

    // After interrupt, slow-path allocation should throw.
    // Use absurdly large size to force cudaMalloc failure and enter slow path.
    CancellationToken token{};
    ASSERT_THROW(CallAllocateDynamic(*pool, 99999999999999ULL, token), ov::Exception);
}

TEST_F(MemoryPoolTest, DynamicAlloc_ReleaseFulfillsPendingRequest) {
    std::unordered_map<BufferID, ptrdiff_t> offsets;
    auto memoryModel = std::make_shared<MemoryModel>(1000, offsets);
    auto pool = std::make_shared<MemoryPool>(1, memoryModel);

    auto allocation = CUDA::DefaultStream::stream().malloc(1024);
    DynamicChunk chunk{allocation, 1024, 0, 1024};

    AddPendingRequest(*pool, 512, 0);
    CallReleaseDynamicChunk(*pool, std::move(chunk));

    ASSERT_EQ(GetPendingRequestCount(*pool), 1u);
    ASSERT_TRUE(IsPendingRequestDone(*pool, 0));
    ASSERT_TRUE(PendingRequestHasChunk(*pool, 0));
    ASSERT_EQ(GetPendingChunkUsableSize(*pool, 0), 512u);
}

TEST_F(MemoryPoolTest, DynamicAlloc_ReleaseSubAllocatesMultiple) {
    std::unordered_map<BufferID, ptrdiff_t> offsets;
    auto memoryModel = std::make_shared<MemoryModel>(1000, offsets);
    auto pool = std::make_shared<MemoryPool>(1, memoryModel);

    const size_t total = 4096;
    auto allocation = CUDA::DefaultStream::stream().malloc(total);
    DynamicChunk chunk{allocation, total, 0, total};

    AddPendingRequest(*pool, 256, 0);
    AddPendingRequest(*pool, 256, 1);

    CallReleaseDynamicChunk(*pool, std::move(chunk));

    ASSERT_EQ(GetPendingRequestCount(*pool), 2u);
    ASSERT_TRUE(IsPendingRequestDone(*pool, 0));
    ASSERT_TRUE(IsPendingRequestDone(*pool, 1));

    // Same base allocation, different offsets
    ASSERT_EQ(GetPendingChunkBasePtr(*pool, 0), GetPendingChunkBasePtr(*pool, 1));
    ASSERT_NE(GetPendingChunkOffset(*pool, 0), GetPendingChunkOffset(*pool, 1));
}

TEST_F(MemoryPoolTest, DynamicAlloc_FIFOHeadServedFirst) {
    std::unordered_map<BufferID, ptrdiff_t> offsets;
    auto memoryModel = std::make_shared<MemoryModel>(1000, offsets);
    auto pool = std::make_shared<MemoryPool>(1, memoryModel);

    const size_t total = 4096;
    auto allocation = CUDA::DefaultStream::stream().malloc(total);
    DynamicChunk chunk{allocation, total, 0, total};

    AddPendingRequest(*pool, 2048, 0);  // head — large
    AddPendingRequest(*pool, 256, 1);   // second — small

    CallReleaseDynamicChunk(*pool, std::move(chunk));

    // Head served first at offset 0
    ASSERT_TRUE(IsPendingRequestDone(*pool, 0));
    ASSERT_EQ(GetPendingChunkOffset(*pool, 0), 0u);
    ASSERT_EQ(GetPendingChunkUsableSize(*pool, 0), 2048u);

    // Second gets remainder
    ASSERT_TRUE(IsPendingRequestDone(*pool, 1));
    ASSERT_GT(GetPendingChunkOffset(*pool, 1), 0u);
    ASSERT_EQ(GetPendingChunkUsableSize(*pool, 1), 256u);
}

TEST_F(MemoryPoolTest, DynamicAlloc_HeadTooLargeTriesMalloc) {
    std::unordered_map<BufferID, ptrdiff_t> offsets;
    auto memoryModel = std::make_shared<MemoryModel>(1000, offsets);
    auto pool = std::make_shared<MemoryPool>(1, memoryModel);

    // Small chunk that won't fit head's request
    auto allocation = CUDA::DefaultStream::stream().malloc(256);
    DynamicChunk chunk{allocation, 256, 0, 256};

    AddPendingRequest(*pool, 512, 0);

    // Chunk freed, then cudaMalloc retried for 512 — should succeed
    CallReleaseDynamicChunk(*pool, std::move(chunk));

    ASSERT_TRUE(IsPendingRequestDone(*pool, 0));
    ASSERT_TRUE(PendingRequestHasChunk(*pool, 0));
    ASSERT_EQ(GetPendingChunkUsableSize(*pool, 0), 512u);
    ASSERT_EQ(GetPendingChunkOffset(*pool, 0), 0u);  // fresh allocation
}

TEST_F(MemoryPoolTest, DynamicAlloc_ReleaseWithNoPendingRequests) {
    std::unordered_map<BufferID, ptrdiff_t> offsets;
    auto memoryModel = std::make_shared<MemoryModel>(1000, offsets);
    auto pool = std::make_shared<MemoryPool>(1, memoryModel);

    auto allocation = CUDA::DefaultStream::stream().malloc(1024);
    DynamicChunk chunk{allocation, 1024, 0, 1024};

    ASSERT_NO_THROW(CallReleaseDynamicChunk(*pool, std::move(chunk)));
    ASSERT_EQ(GetPendingRequestCount(*pool), 0u);
}

TEST_F(MemoryPoolTest, DynamicAlloc_ThreadedAllocAndRelease) {
    CancellationToken token{};
    std::unordered_map<BufferID, ptrdiff_t> offsets;
    auto memoryModel = std::make_shared<MemoryModel>(1000, offsets);
    auto pool = std::make_shared<MemoryPool>(1, memoryModel);
    auto proxy = pool->WaitAndGet(token);

    constexpr int kThreadCount = 4;
    constexpr int kAllocsPerThread = 10;
    std::vector<std::thread> threads;

    for (int t = 0; t < kThreadCount; ++t) {
        threads.emplace_back([&proxy, &token]() {
            for (int i = 0; i < kAllocsPerThread; ++i) {
                auto handle = proxy.AllocateDynamic(256, token);
                ASSERT_NE(handle.get(), nullptr);
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }
}

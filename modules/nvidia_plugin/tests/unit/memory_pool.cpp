// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "memory_manager/cuda_memory_pool.hpp"
#include "memory_manager/model/cuda_memory_model.hpp"

using namespace ov::nvidia_gpu;

class MemoryPoolTest : public testing::Test {
    void SetUp() override {}

    void TearDown() override {}

public:
    size_t GetNumAvailableMemoryManagers(MemoryPool& memManPool) { return memManPool.memory_blocks_.size(); }
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

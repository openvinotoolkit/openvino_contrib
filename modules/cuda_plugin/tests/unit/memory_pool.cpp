// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <typeinfo>
#include <condition_variable>
#include <gtest/gtest.h>

#include <ngraph/function.hpp>
#include <ngraph/node.hpp>
#include <cuda_plugin.hpp>
#include <cuda_executable_network.hpp>
#include <memory_manager/model/cuda_memory_model.hpp>
#include <threading/ie_executor_manager.hpp>

using namespace CUDAPlugin;

class MemoryManagerPoolTest : public testing::Test {
    void SetUp() override {
    }

    void TearDown() override {
    }

 public:
    size_t GetNumAvailableMemoryManagers(MemoryManagerPool& memManPool) {
        return memManPool.memory_managers_.size();
    }
};

TEST_F(MemoryManagerPoolTest, MemoryManagerProxy_Success) {
    CancellationToken cancellationToken{};
    std::unordered_map<MemoryModel::TensorID, ptrdiff_t> offsets;
    auto memoryModel = std::make_shared<MemoryModel>(1000, offsets);
    auto memoryPool = std::make_shared<MemoryManagerPool>(2, nullptr, memoryModel);
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

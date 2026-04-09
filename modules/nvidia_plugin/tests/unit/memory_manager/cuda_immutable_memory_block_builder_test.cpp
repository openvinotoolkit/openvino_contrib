// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "memory_manager/cuda_immutable_memory_block_builder.hpp"

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <vector>

#include "memory_manager/cuda_device_mem_block.hpp"
#include "openvino/core/except.hpp"

TEST(ImmutableMemoryBlockBuilder, BuildEmpty) {
    using namespace ov::nvidia_gpu;

    ImmutableMemoryBlockBuilder builder;
    EXPECT_NO_THROW(builder.build());
}

TEST(ImmutableMemoryBlockBuilder, Build) {
    using namespace ov::nvidia_gpu;

    const BufferID t0_id = 1;
    const std::vector<uint8_t> t0_data(16, 0xA5);
    const BufferID t1_id = 3;
    const std::vector<uint8_t> t1_data(4792, 0x5A);
    const BufferID t2_id = 5;
    const std::vector<uint8_t> t2_data(798, 0xC3);

    std::shared_ptr<DeviceMemBlock> memory_block;
    {
        ImmutableMemoryBlockBuilder builder;
        MemoryModel::Ptr memoryModel;
        builder.addAllocation(t0_id, &t0_data[0], t0_data.size());
        builder.addAllocation(t1_id, &t1_data[0], t1_data.size());
        builder.addAllocation(t2_id, &t2_data[0], t2_data.size());
        ASSERT_NO_THROW({ std::tie(memory_block, memoryModel) = builder.build(); });
    }

    auto verify_device_data = [](void* device_ptr, const std::vector<uint8_t>& expected_data) {
        ASSERT_TRUE(device_ptr != nullptr);
        std::vector<uint8_t> data_from_device(expected_data.size(), 0);
        auto err = ::cudaMemcpy(&data_from_device[0], device_ptr, expected_data.size(), cudaMemcpyDeviceToHost);
        ASSERT_EQ(err, cudaSuccess);
        ASSERT_EQ(data_from_device, expected_data);
    };

    verify_device_data(memory_block->deviceTensorPtr(TensorID{t0_id}), t0_data);
    verify_device_data(memory_block->deviceTensorPtr(TensorID{t1_id}), t1_data);
    verify_device_data(memory_block->deviceTensorPtr(TensorID{t2_id}), t2_data);
}

TEST(ImmutableMemoryBlockBuilder, HandleDuplicateAllocation) {
    using namespace ov::nvidia_gpu;

    ImmutableMemoryBlockBuilder builder;

    BufferID duplicate_buffer_id = 1;
    const std::vector<uint8_t> t0_data(16, 0xA5);
    const std::vector<uint8_t> t1_data(32, 0xA5);

    builder.addAllocation(duplicate_buffer_id, &t0_data[0], t0_data.size());

    ASSERT_THROW(builder.addAllocation(duplicate_buffer_id, &t0_data[0], t0_data.size()), ov::Exception);
    ASSERT_THROW(builder.addAllocation(duplicate_buffer_id, &t1_data[0], t1_data.size()), ov::Exception);
}

TEST(ImmutableMemoryBlockBuilder, HandleZeroAllocationSize) {
    using namespace ov::nvidia_gpu;

    ImmutableMemoryBlockBuilder builder;

    BufferID buffer_id = 1;
    const std::vector<uint8_t> data(16, 0xA5);

    ASSERT_THROW(builder.addAllocation(buffer_id, &data[0], 0), ov::Exception);
    ASSERT_THROW(builder.addAllocation(buffer_id, nullptr, 0), ov::Exception);
}

TEST(ImmutableMemoryBlockBuilder, HandleNullDataPointer) {
    using namespace ov::nvidia_gpu;

    ImmutableMemoryBlockBuilder builder;

    BufferID buffer_id = 1;

    ASSERT_THROW(builder.addAllocation(buffer_id, nullptr, 128), ov::Exception);
}

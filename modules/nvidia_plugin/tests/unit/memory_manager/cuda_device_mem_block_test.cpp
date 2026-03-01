// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "memory_manager/cuda_device_mem_block.hpp"

#include <gtest/gtest.h>

#include "memory_manager/model/cuda_memory_model.hpp"

TEST(DeviceMemBlock, ZeroSizeMemoryBlock) {
    using namespace ov::nvidia_gpu;

    // "No constant tensors" edge case. MemoryModel is empty and
    // memory block size is zero.
    {
        const size_t bsize = 0;
        std::unordered_map<BufferID, ptrdiff_t> offsets;
        auto model = std::make_shared<MemoryModel>(bsize, offsets);
        ASSERT_EQ(model->deviceMemoryBlockSize(), 0);

        auto mem_block = std::make_unique<DeviceMemBlock>(model);
        ASSERT_TRUE(mem_block->deviceTensorPtr(TensorID{0}) == nullptr);
    }

    // The only allocation is of zero size. Zero size tensors are forbidden
    // by MemoryModel builder's implementations and this will never happen
    // in runtime for static models.
    // For dynamic-only models, the block size may be 0 but DeviceMemBlock
    // still allocates at least 1 byte to ensure a valid device pointer.
    {
        const BufferID buffer_id = 1;
        const ptrdiff_t offset = 0;
        const std::unordered_map<BufferID, ptrdiff_t> offsets = {{buffer_id, offset}};
        const size_t bsize = 0;
        auto model = std::make_shared<MemoryModel>(bsize, offsets);
        ASSERT_EQ(model->deviceMemoryBlockSize(), 0);

        auto mem_block = std::make_unique<DeviceMemBlock>(model);
        // DeviceMemBlock allocates max(size, 1) bytes, so pointer is valid
        ASSERT_TRUE(mem_block->deviceTensorPtr(TensorID{buffer_id}) != nullptr);
    }
}

TEST(DeviceMemBlock, VerifyDevicePointers) {
    using namespace ov::nvidia_gpu;

    const BufferID alloc_count = 5;
    const size_t allocation_size = 0x100;
    std::unordered_map<BufferID, ptrdiff_t> offsets;
    for (BufferID id = 0; id < alloc_count; ++id) {
        offsets[id] = id * allocation_size;
    }
    const size_t block_size = alloc_count * allocation_size;

    auto model = std::make_shared<MemoryModel>(block_size, offsets);
    auto mem_block = std::make_unique<DeviceMemBlock>(model);

    const BufferID first_allocation_id = 0;
    const uint8_t* const block_base_addr =
        reinterpret_cast<uint8_t*>(mem_block->deviceTensorPtr(TensorID{first_allocation_id}));
    ASSERT_TRUE(block_base_addr != nullptr);

    // Verify tensor pointers
    for (BufferID id = 0; id < alloc_count; ++id) {
        const uint8_t* const actual_addr = reinterpret_cast<uint8_t*>(mem_block->deviceTensorPtr(TensorID{id}));
        const uint8_t* const expected_addr = block_base_addr + offsets.at(id);
        ASSERT_EQ(actual_addr, expected_addr);
    }
}

TEST(DeviceMemBlock, NullPtrIfTensorNotFound) {
    using namespace ov::nvidia_gpu;

    const size_t block_size = 0x700;
    std::unordered_map<BufferID, ptrdiff_t> offsets = {
        {1, 0x0},
        {2, 0x300},
        {3, 0x500},
    };
    auto model = std::make_shared<MemoryModel>(block_size, offsets);
    auto mem_block = std::make_unique<DeviceMemBlock>(model);

    // Returns nullptr for unknown tensor id's
    ASSERT_TRUE(mem_block->deviceTensorPtr(TensorID{0}) == nullptr);
    ASSERT_TRUE(mem_block->deviceTensorPtr(TensorID{4}) == nullptr);
}

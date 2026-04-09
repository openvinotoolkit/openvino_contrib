// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "memory_manager/model/cuda_memory_model.hpp"

#include <gtest/gtest.h>

TEST(MemoryModel, Empty) {
    using namespace ov::nvidia_gpu;

    constexpr size_t bsize = 0;

    std::unordered_map<BufferID, ptrdiff_t> offsets;
    MemoryModel::Ptr model = std::make_shared<MemoryModel>(bsize, offsets);

    ASSERT_EQ(model->deviceMemoryBlockSize(), 0);

    ptrdiff_t offset{};
    ASSERT_FALSE(model->offsetForBuffer(0, offset));
    ASSERT_FALSE(model->offsetForBuffer(1, offset));
}

TEST(MemoryModel, NotEmpty) {
    using namespace ov::nvidia_gpu;

    constexpr size_t bsize = 0x354700;

    BufferID invalid_id = 0, id1 = 1, id2 = 2;
    ptrdiff_t offset1 = 0, offset2 = 0x254000;
    std::unordered_map<BufferID, ptrdiff_t> offsets = {
        {id1, offset1},
        {id2, offset2},
    };

    MemoryModel::Ptr model = std::make_shared<MemoryModel>(bsize, offsets);

    ASSERT_EQ(model->deviceMemoryBlockSize(), bsize);

    ptrdiff_t offset{};

    ASSERT_FALSE(model->offsetForBuffer(invalid_id, offset));

    ASSERT_TRUE(model->offsetForBuffer(id1, offset));
    ASSERT_EQ(offset, offset1);

    ASSERT_TRUE(model->offsetForBuffer(id2, offset));
    ASSERT_EQ(offset, offset2);
}

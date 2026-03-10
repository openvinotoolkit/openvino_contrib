// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "memory_manager/model/cuda_immutable_memory_model_builder.hpp"

#include <gtest/gtest.h>

#include "memory_manager/model/details/cuda_memory_utils.hpp"
#include "openvino/core/except.hpp"

TEST(ImmutableMemoryModelBuilder, BuildEmpty) {
    using namespace ov::nvidia_gpu;

    ImmutableMemoryModelBuilder builder;
    MemoryModel::Ptr model = builder.build();
    ASSERT_EQ(model->deviceMemoryBlockSize(), 0);
}

TEST(ImmutableMemoryModelBuilder, Build) {
    using namespace ov::nvidia_gpu;

    BufferID id1 = 0;
    size_t size1 = 1;
    BufferID id2 = 3;
    size_t size2 = 256;
    BufferID id3 = 7;
    size_t size3 = 784;
    BufferID id4 = 9;
    size_t size4 = 1396;

    ImmutableMemoryModelBuilder builder;
    builder.addAllocation(id1, size1);
    builder.addAllocation(id2, size2);
    builder.addAllocation(id3, size3);
    builder.addAllocation(id4, size4);
    MemoryModel::Ptr model = builder.build();

    ptrdiff_t offset1 = -1;
    ASSERT_TRUE(model->offsetForBuffer(id1, offset1));
    ASSERT_EQ(offset1, 0);

    ptrdiff_t offset2 = -1;
    ASSERT_TRUE(model->offsetForBuffer(id2, offset2));
    ASSERT_EQ(offset2, offset1 + applyAllignment(size1));

    ptrdiff_t offset3 = -1;
    ASSERT_TRUE(model->offsetForBuffer(id3, offset3));
    ASSERT_EQ(offset3, offset2 + applyAllignment(size2));

    ptrdiff_t offset4 = -1;
    ASSERT_TRUE(model->offsetForBuffer(id4, offset4));
    ASSERT_EQ(offset4, offset3 + applyAllignment(size3));

    ASSERT_EQ(model->deviceMemoryBlockSize(), offset4 + applyAllignment(size4));
}

TEST(ImmutableMemoryModelBuilder, HandleDuplicateAllocation) {
    using namespace ov::nvidia_gpu;

    ImmutableMemoryModelBuilder builder;

    BufferID duplicate_buffer_id = 1;
    size_t size1 = 128;
    size_t size2 = 256;

    builder.addAllocation(duplicate_buffer_id, size1);

    ASSERT_THROW(builder.addAllocation(duplicate_buffer_id, size1), ov::Exception);
    ASSERT_THROW(builder.addAllocation(duplicate_buffer_id, size2), ov::Exception);
}

TEST(ImmutableMemoryModelBuilder, HandleZeroAllocationSize) {
    using namespace ov::nvidia_gpu;

    ImmutableMemoryModelBuilder builder;

    BufferID buffer_id = 1;

    ASSERT_THROW(builder.addAllocation(buffer_id, 0), ov::Exception);
}
